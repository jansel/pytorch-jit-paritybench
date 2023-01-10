import sys
_module = sys.modules[__name__]
del sys
data = _module
data_sampler = _module
data_util = _module
ffhq_dataset = _module
paired_image_dataset = _module
prefetch_dataloader = _module
reds_dataset = _module
single_image_dataset = _module
transforms = _module
video_test_dataset = _module
vimeo90k_dataset = _module
metrics = _module
fid = _module
metric_util = _module
niqe = _module
psnr_ssim = _module
models = _module
archs = _module
arch_util = _module
dfdnet_arch = _module
dfdnet_util = _module
discriminator_arch = _module
duf_arch = _module
edsr_arch = _module
edvr_arch = _module
inception = _module
rcan_arch = _module
rrdbnet_arch = _module
spynet_arch = _module
srresnet_arch = _module
stylegan2_arch = _module
tof_arch = _module
vgg_arch = _module
base_model = _module
edvr_model = _module
esrgan_model = _module
losses = _module
loss_util = _module
losses = _module
lr_scheduler = _module
dcn = _module
deform_conv = _module
fused_act = _module
fused_act = _module
upfirdn2d = _module
upfirdn2d = _module
sr_model = _module
srgan_model = _module
stylegan2_model = _module
video_base_model = _module
video_gan_model = _module
test = _module
train = _module
utils = _module
dist_util = _module
download_util = _module
face_util = _module
file_client = _module
flow_util = _module
img_util = _module
lmdb_util = _module
logger = _module
matlab_functions = _module
misc = _module
options = _module
inference_dfdnet = _module
inference_esrgan = _module
inference_stylegan2 = _module
create_lmdb = _module
download_datasets = _module
extract_images_from_tfrecords = _module
extract_subimages = _module
generate_meta_info = _module
regroup_reds_dataset = _module
download_gdrive = _module
download_pretrained_models = _module
calculate_fid_folder = _module
calculate_fid_stats_from_datasets = _module
calculate_lpips = _module
calculate_psnr_ssim = _module
calculate_stylegan2_fid = _module
convert_dfdnet = _module
convert_models = _module
convert_stylegan = _module
publish_models = _module
setup = _module
test_discriminator_backward = _module
test_ffhq_dataset = _module
test_lr_scheduler = _module
test_niqe = _module
test_paired_image_dataset = _module
test_reds_dataset = _module
test_vimeo90k_dataset = _module

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


import numpy as np


import random


import torch


import torch.utils.data


from functools import partial


import math


from torch.utils.data.sampler import Sampler


from torch.nn import functional as F


from torch.utils import data as data


from torchvision.transforms.functional import normalize


import queue as Queue


from torch.utils.data import DataLoader


import torch.nn as nn


from scipy import linalg


from torch import nn as nn


from torch.nn import init as init


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.functional as F


import torch.nn.utils.spectral_norm as SpectralNorm


from torch.autograd import Function


from torch.utils.model_zoo import load_url


from torchvision import models


from torch import nn


from collections import OrderedDict


from torchvision.models import vgg as vgg


import logging


from copy import deepcopy


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import functools


from torch import autograd as autograd


from collections import Counter


from torch.optim.lr_scheduler import _LRScheduler


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


from torch import distributed as dist


import time


import torch.distributed as dist


import torch.multiprocessing as mp


from torchvision.utils import make_grid


import torchvision.transforms as transforms


from torchvision import utils


from torch.serialization import _is_zipfile


from torch.serialization import _open_file_like


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import copy


import torchvision.utils


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class BlurFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        grad_input = F.conv2d(grad_output, kernel_flip, padding=1, groups=grad_output.shape[1])
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = F.conv2d(gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1])
        return grad_input, None, None


class BlurFunction(Function):

    @staticmethod
    def forward(ctx, x, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)
        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):

    def __init__(self, channel):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel / kernel.sum()
        kernel_flip = torch.flip(kernel, [2, 3])
        self.kernel = kernel.repeat(channel, 1, 1, 1)
        self.kernel_flip = kernel_flip.repeat(channel, 1, 1, 1)

    def forward(self, x):
        return blur(x, self.kernel.type_as(x), self.kernel_flip.type_as(x))


class SFTUpBlock(nn.Module):
    """Spatial feature transform (SFT) with upsampling block."""

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(SFTUpBlock, self).__init__()
        self.conv1 = nn.Sequential(Blur(in_channel), SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)), nn.LeakyReLU(0.04, True))
        self.convup = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)), nn.LeakyReLU(0.2, True))
        self.scale_block = nn.Sequential(SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)))
        self.shift_block = nn.Sequential(SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.Sigmoid())

    def forward(self, x, updated_feat):
        out = self.conv1(x)
        scale = self.scale_block(updated_feat)
        shift = self.shift_block(updated_feat)
        out = out * scale + shift
        out = self.convup(out)
        return out


def AttentionBlock(in_channel):
    return nn.Sequential(SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)))


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """Conv block used in MSDilationBlock."""
    return nn.Sequential(SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation, bias=bias)), nn.LeakyReLU(0.2), SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation, bias=bias)))


class MSDilationBlock(nn.Module):
    """Multi-scale dilation block."""

    def __init__(self, in_channels, kernel_size=3, dilation=[1, 1, 1, 1], bias=True):
        super(MSDilationBlock, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(4):
            self.conv_blocks.append(conv_block(in_channels, in_channels, kernel_size, dilation=dilation[i], bias=bias))
        self.conv_fusion = SpectralNorm(nn.Conv2d(in_channels * 4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias))

    def forward(self, x):
        out = []
        for i in range(4):
            out.append(self.conv_blocks[i](x))
        out = torch.cat(out, 1)
        out = self.conv_fusion(out) + x
        return out


class UpResBlock(nn.Module):

    def __init__(self, in_channel):
        super(UpResBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(in_channel, in_channel, 3, 1, 1))

    def forward(self, x):
        out = x + self.body(x)
        return out


NAMES = {'vgg11': ['conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg13': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg16': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], 'vgg19': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']}


VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, range_norm=False, requires_grad=False, remove_pooling=False, pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)
        features = vgg_net.features[:max_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v
        self.vgg_net = nn.Sequential(modified_net)
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output


def calc_mean_std(feat, eps=1e-05):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class DFDNet(nn.Module):
    """DFDNet: Deep Face Dictionary Network.

    It only processes faces with 512x512 size.
    """

    def __init__(self, num_feat, dict_path):
        super().__init__()
        self.parts = ['left_eye', 'right_eye', 'nose', 'mouth']
        channel_sizes = [128, 256, 512, 512]
        self.feature_sizes = np.array([256, 128, 64, 32])
        self.vgg_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'conv5_4']
        self.flag_dict_device = False
        self.dict = torch.load(dict_path)
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=self.vgg_layers, vgg_type='vgg19', use_input_norm=True, range_norm=True, requires_grad=False)
        self.attn_blocks = nn.ModuleDict()
        for idx, feat_size in enumerate(self.feature_sizes):
            for name in self.parts:
                self.attn_blocks[f'{name}_{feat_size}'] = AttentionBlock(channel_sizes[idx])
        self.multi_scale_dilation = MSDilationBlock(num_feat * 8, dilation=[4, 3, 2, 1])
        self.upsample0 = SFTUpBlock(num_feat * 8, num_feat * 8)
        self.upsample1 = SFTUpBlock(num_feat * 8, num_feat * 4)
        self.upsample2 = SFTUpBlock(num_feat * 4, num_feat * 2)
        self.upsample3 = SFTUpBlock(num_feat * 2, num_feat)
        self.upsample4 = nn.Sequential(SpectralNorm(nn.Conv2d(num_feat, num_feat, 3, 1, 1)), nn.LeakyReLU(0.2, True), UpResBlock(num_feat), UpResBlock(num_feat), nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def swap_feat(self, vgg_feat, updated_feat, dict_feat, location, part_name, f_size):
        """swap the features from the dictionary."""
        part_feat = vgg_feat[:, :, location[1]:location[3], location[0]:location[2]].clone()
        part_resize_feat = F.interpolate(part_feat, dict_feat.size()[2:4], mode='bilinear', align_corners=False)
        dict_feat = adaptive_instance_normalization(dict_feat, part_resize_feat)
        similarity_score = F.conv2d(part_resize_feat, dict_feat)
        similarity_score = F.softmax(similarity_score.view(-1), dim=0)
        select_idx = torch.argmax(similarity_score)
        swap_feat = F.interpolate(dict_feat[select_idx:select_idx + 1], part_feat.size()[2:4])
        attn = self.attn_blocks[f'{part_name}_' + str(f_size)](swap_feat - part_feat)
        attn_feat = attn * swap_feat
        updated_feat[:, :, location[1]:location[3], location[0]:location[2]] = attn_feat + part_feat
        return updated_feat

    def put_dict_to_device(self, x):
        if self.flag_dict_device is False:
            for k, v in self.dict.items():
                for kk, vv in v.items():
                    self.dict[k][kk] = vv
            self.flag_dict_device = True

    def forward(self, x, part_locations):
        """
        Now only support testing with batch size = 0.

        Args:
            x (Tensor): Input faces with shape (b, c, 512, 512).
            part_locations (list[Tensor]): Part locations.
        """
        self.put_dict_to_device(x)
        vgg_features = self.vgg_extractor(x)
        updated_vgg_features = []
        batch = 0
        for vgg_layer, f_size in zip(self.vgg_layers, self.feature_sizes):
            dict_features = self.dict[f'{f_size}']
            vgg_feat = vgg_features[vgg_layer]
            updated_feat = vgg_feat.clone()
            for part_idx, part_name in enumerate(self.parts):
                location = (part_locations[part_idx][batch] // (512 / f_size)).int()
                updated_feat = self.swap_feat(vgg_feat, updated_feat, dict_features[part_name], location, part_name, f_size)
            updated_vgg_features.append(updated_feat)
        vgg_feat_dilation = self.multi_scale_dilation(vgg_features['conv5_4'])
        upsampled_feat = self.upsample0(vgg_feat_dilation, updated_vgg_features[3])
        upsampled_feat = self.upsample1(upsampled_feat, updated_vgg_features[2])
        upsampled_feat = self.upsample2(upsampled_feat, updated_vgg_features[1])
        upsampled_feat = self.upsample3(upsampled_feat, updated_vgg_features[0])
        out = self.upsample4(upsampled_feat)
        return out


class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator128, self).__init__()
        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)
        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, f'Input spatial size must be 128x128, but received {x.size()}.'
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


class DenseBlocksTemporalReduce(nn.Module):
    """A concatenation of 3 dense blocks with reduction in temporal dimension.

    Note that the output temporal dimension is 6 fewer the input temporal
    dimension, since there are 3 blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32
        adapt_official_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, adapt_official_weights=False):
        super(DenseBlocksTemporalReduce, self).__init__()
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.temporal_reduce1 = nn.Sequential(nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat, num_feat, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
        self.temporal_reduce2 = nn.Sequential(nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + num_grow_ch, num_feat + num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
        self.temporal_reduce3 = nn.Sequential(nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + 2 * num_grow_ch, num_feat + 2 * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_grow_ch * 3, 1, h, w).
        """
        x1 = self.temporal_reduce1(x)
        x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        x2 = self.temporal_reduce2(x1)
        x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        x3 = self.temporal_reduce3(x2)
        x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        return x3


class DenseBlocks(nn.Module):
    """ A concatenation of N dense blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32.
        num_block (int): Number of dense blocks. The values are:
            DUF-S (16 layers): 3
            DUF-M (18 layers): 9
            DUF-L (52 layers): 21
        adapt_official_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, num_block, num_feat=64, num_grow_ch=16, adapt_official_weights=False):
        super(DenseBlocks, self).__init__()
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(nn.Sequential(nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + i * num_grow_ch, num_feat + i * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape
                (b, num_feat + num_block * num_grow_ch, t, h, w).
        """
        for i in range(0, len(self.dense_blocks)):
            y = self.dense_blocks[i](x)
            x = torch.cat((x, y), 1)
        return x


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Ref: https://github.com/yhjo09/VSR-DUF.
    It only supports input with 3 channels. And it applies the same filters
    to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters.
            The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError(f'The type of filter_size must be tuple, but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError(f'The length of filter size must be 2, but got {len(filter_size)}.')
        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        expansion_filter = torch.eye(int(filter_prod)).view(filter_prod, 1, *filter_size)
        self.expansion_filter = expansion_filter.repeat(3, 1, 1, 1)

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            filters (Tensor): Generated dynamic filters.
                The shape is (n, filter_prod, upsampling_square, h, w).
                filter_prod: prod of filter kenrel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle,
                    upsampling_square = upsampling * upsampling
                    e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            Tensor: Filtered image with shape (n, 3*upsampling_square, h, w)
        """
        n, filter_prod, upsampling_square, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = F.conv2d(x, self.expansion_filter, padding=(kh // 2, kw // 2), groups=3)
        expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(0, 3, 4, 1, 2)
        filters = filters.permute(0, 3, 4, 1, 2)
        out = torch.matmul(expanded_input, filters)
        return out.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)


class DUF(nn.Module):
    """Network architecture for DUF

    Paper: Jo et.al. Deep Video Super-Resolution Network Using Dynamic
            Upsampling Filters Without Explicit Motion Compensation, CVPR, 2018
    Code reference:
        https://github.com/yhjo09/VSR-DUF
    For all the models below, 'adapt_official_weights' is only necessary when
    loading the weights converted from the official TensorFlow weights.
    Please set it to False if you are training the model from scratch.

    There are three models with different model size: DUF16Layers, DUF28Layers,
    and DUF52Layers. This class is the base class for these models.

    Args:
        scale (int): The upsampling factor. Default: 4.
        num_layer (int): The number of layers. Default: 52.
        adapt_official_weights_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, scale=4, num_layer=52, adapt_official_weights=False):
        super(DUF, self).__init__()
        self.scale = scale
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.conv3d1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dynamic_filter = DynamicUpsamplingFilter((5, 5))
        if num_layer == 16:
            num_block = 3
            num_grow_ch = 32
        elif num_layer == 28:
            num_block = 9
            num_grow_ch = 16
        elif num_layer == 52:
            num_block = 21
            num_grow_ch = 16
        else:
            raise ValueError(f'Only supported (16, 28, 52) layers, but got {num_layer}.')
        self.dense_block1 = DenseBlocks(num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch, adapt_official_weights=adapt_official_weights)
        self.dense_block2 = DenseBlocksTemporalReduce(64 + num_grow_ch * num_block, num_grow_ch, adapt_official_weights=adapt_official_weights)
        channels = 64 + num_grow_ch * num_block + num_grow_ch * 3
        self.bn3d2 = nn.BatchNorm3d(channels, eps=eps, momentum=momentum)
        self.conv3d2 = nn.Conv3d(channels, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * scale ** 2, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * scale ** 2, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape (b, 7, c, h, w)

        Returns:
            Tensor: Output with shape (b, 1, h * scale, w * scale)
        """
        num_batches, num_imgs, _, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x_center = x[:, :, num_imgs // 2, :, :]
        x = self.conv3d1(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = F.relu(self.bn3d2(x), inplace=True)
        x = F.relu(self.conv3d2(x), inplace=True)
        res = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))
        filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))
        filter_ = F.softmax(filter_.view(num_batches, 25, self.scale ** 2, h, w), dim=1)
        out = self.dynamic_filter(x_center, filter_)
        out += res.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)
        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=16, upscale=4, res_scale=1, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.404)):
        super(EDSR, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_ext.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_ext.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        feat = self.lrelu(self.feat_fusion(aligned_feat))
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)
        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)
        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))
        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))
        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5, deformable_groups=8, num_extract_block=5, num_reconstruct_block=10, center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True):
        super(EDVR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, 'The height and width must be multiple of 16.'
        else:
            assert h % 4 == 0 and w % 4 == 0, 'The height and width must be multiple of 4.'
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)
        ref_feat_l = [feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(), feat_l3[:, self.center_frame_idx, :, :, :].clone()]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


LOCAL_FID_WEIGHTS = 'experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation.

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    try:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False, init_weights=False)
    except TypeError:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    if os.path.exists(LOCAL_FID_WEIGHTS):
        state_dict = torch.load(LOCAL_FID_WEIGHTS, map_location=lambda storage, loc: storage)
    else:
        state_dict = load_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3.

        Args:
            output_blocks (list[int]): Indices of blocks to return features of.
                Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
            resize_input (bool): If true, bilinearly resizes input to width and
                height 299 before feeding input to model. As the network
                without fully connected layers is fully convolutional, it
                should be able to handle inputs of arbitrary size, so resizing
                might not be strictly needed. Default: True.
            normalize_input (bool): If true, scales the input from range (0, 1)
                to the range the pretrained Inception network expects,
                namely (-1, 1). Default: True.
            requires_grad (bool): If true, parameters of the model require
                gradients. Possibly useful for finetuning the network.
                Default: False.
            use_fid_inception (bool): If true, uses the pretrained Inception
                model used in Tensorflow's FID implementation.
                If false, uses the pretrained Inception model available in
                torchvision. The FID Inception model has different weights
                and a slightly different structure from torchvision's
                Inception model. If you want to compute FID scores, you are
                strongly advised to set this parameter to true to get
                comparable results. Default: True.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            try:
                inception = models.inception_v3(pretrained=True, init_weights=False)
            except TypeError:
                inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """Get Inception feature maps.

        Args:
            x (Tensor): Input tensor of shape (b, 3, h, w).
                Values are expected to be in range (-1, 1). You can also input
                (0, 1) with setting normalize_input = True.

        Returns:
            list[Tensor]: Corresponding to the selected output block, sorted
            ascending by index.
        """
        output = []
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)
            if idx == self.last_needed_block:
                break
        return output


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale
        self.rcab = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()
        self.residual_group = make_layer(RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_group=10, num_block=16, squeeze_factor=16, upscale=4, res_scale=1, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.404)):
        super(RCAN, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualGroup, num_group, num_feat=num_feat, num_block=num_block, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class BasicModule(nn.Module):
    """Basic module of SPyNet.

    Note that unlike the architecture in spynet_arch.py, the basic module
    here contains batch normalization.
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Estimated flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = ref[0].new_zeros([ref[0].size(0), 2, int(math.floor(ref[0].size(2) / 2.0)), int(math.floor(ref[0].size(3) / 2.0))])
        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')
            flow = self.basic_module[level](torch.cat([ref[level], flow_warp(supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'), upsampled_flow], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()
        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)
        return flow


class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = arch_util.make_layer(arch_util.ResidualBlockNoBN, num_block, num_feat=num_feat)
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        arch_util.default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            arch_util.default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_ext.upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_ext.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_ext.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    else:
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out


class UpFirDnUpsample(nn.Module):
    """Upsample, FIR filter, and downsample (upsampole version).

    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * factor ** 2
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2 + factor - 1, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnDownsample(nn.Module):
    """Upsample, FIR filter, and downsample (downsampole version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * upsample_factor ** 2
        if upsample_factor > 1:
            pad = self.kernel.shape[0] - upsample_factor - (kernel_size - 1)
            self.pad = (pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1
        elif downsample_factor > 1:
            pad = self.kernel.shape[0] - downsample_factor + (kernel_size - 1)
            self.pad = (pad + 1) // 2, pad // 2
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}, downsample_factor={self.downsample_factor})'


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused_act_ext.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused_act_ext.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused_act_ext.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f"Wrong activation value in EqualLinear: {activation}Supported ones are: ['fused_lrelu', None].")
        self.scale = 1 / math.sqrt(in_channels) * lr_mul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})'


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None, resample_kernel=(1, 3, 3, 1), eps=1e-08):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(resample_kernel, upsample_factor=2, downsample_factor=1, kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(f"Wrong sample mode {self.sample_mode}, supported ones are ['upsample', 'downsample', None].")
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.modulation = EqualLinear(num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.sample_mode == 'upsample':
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            x = x.view(1, b * c, h, w)
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})'


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class StyleConv(nn.Module):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None, resample_kernel=(1, 3, 3, 1)):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode, resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, resample_kernel=(1, 3, 3, 1)):
        super(ToRGB, self).__init__()
        if upsample:
            self.upsample = UpFirDnUpsample(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.modulated_conv = ModulatedConv2d(in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), lr_mlp=0.01, narrow=1):
        super(StyleGAN2Generator, self).__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(EqualLinear(num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        channels = {'4': int(512 * narrow), '8': int(512 * narrow), '16': int(512 * narrow), '32': int(512 * narrow), '64': int(256 * channel_multiplier * narrow), '128': int(128 * channel_multiplier * narrow), '256': int(64 * channel_multiplier * narrow), '512': int(32 * channel_multiplier * narrow), '1024': int(16 * channel_multiplier * narrow)}
        self.channels = channels
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(channels['4'], channels['4'], kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False, resample_kernel=resample_kernel)
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channels = channels['4']
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.style_convs.append(StyleConv(in_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode='upsample', resample_kernel=resample_kernel))
            self.style_convs.append(StyleConv(out_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self, styles, input_is_latent=False, noise=None, randomize_noise=True, truncation=1, truncation_latent=None, inject_index=None, return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})'


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True):
        layers = []
        if downsample:
            layers.append(UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        layers.append(EqualConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias and not activate))
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1)):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(in_channels, out_channels, 3, downsample=True, resample_kernel=resample_kernel, bias=True, activate=True)
        self.skip = ConvLayer(in_channels, out_channels, 1, downsample=True, resample_kernel=resample_kernel, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator, self).__init__()
        channels = {'4': int(512 * narrow), '8': int(512 * narrow), '16': int(512 * narrow), '32': int(512 * narrow), '64': int(256 * channel_multiplier * narrow), '128': int(128 * channel_multiplier * narrow), '256': int(64 * channel_multiplier * narrow), '512': int(32 * channel_multiplier * narrow), '1024': int(16 * channel_multiplier * narrow)}
        log_size = int(math.log(out_size, 2))
        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]
        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2 ** (i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)
        self.final_conv = ConvLayer(in_channels + 1, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(EqualLinear(channels['4'] * 4 * 4, channels['4'], bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'), EqualLinear(channels['4'], 1, bias=True, bias_init_val=0, lr_mul=1, activation=None))
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)
        b, c, h, w = out.shape
        group = min(b, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)
        return out


class SPyNetTOF(nn.Module):
    """SPyNet architecture for TOF.

    Note that this implementation is specifically for TOFlow. Please use
    spynet_arch.py for general use. They differ in the following aspects:
        1. The basic modules here contain BatchNorm.
        2. Normalization and denormalization are not done here, as
            they are done in TOFlow.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network
    Code reference:
        https://github.com/Coldog2333/pytoflow

    Args:
        load_path (str): Path for pretrained SPyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SPyNetTOF, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(4)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, ref, supp):
        """
        Args:
            ref (Tensor): Reference image with shape of (b, 3, h, w).
            supp: The supporting image to be warped: (b, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (b, 2, h, w).
        """
        num_batches, _, h, w = ref.size()
        ref = [ref]
        supp = [supp]
        for _ in range(3):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = ref[0].new_zeros(num_batches, 2, h // 16, w // 16)
        for i in range(4):
            flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[i](torch.cat([ref[i], flow_warp(supp[i], flow_up.permute(0, 2, 3, 1)), flow_up], 1))
        return flow


class TOFlow(nn.Module):
    """PyTorch implementation of TOFlow.

    In TOFlow, the LR frames are pre-upsampled and have the same size with
    the GT frames.
    Paper:
        Xue et al., Video Enhancement with Task-Oriented Flow, IJCV 2018
    Code reference:
        1. https://github.com/anchen1011/toflow
        2. https://github.com/Coldog2333/pytoflow

    Args:
        adapt_official_weights (bool): Whether to adapt the weights translated
            from the official implementation. Set to false if you want to
            train from scratch. Default: False
    """

    def __init__(self, adapt_official_weights=False):
        super(TOFlow, self).__init__()
        self.adapt_official_weights = adapt_official_weights
        self.ref_idx = 0 if adapt_official_weights else 3
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.spynet = SPyNetTOF()
        self.conv_1 = nn.Conv2d(3 * 7, 64, 9, 1, 4)
        self.conv_2 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_3 = nn.Conv2d(64, 64, 1)
        self.conv_4 = nn.Conv2d(64, 3, 1)
        self.relu = nn.ReLU(inplace=True)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, lrs):
        """
        Args:
            lrs: Input lr frames: (b, 7, 3, h, w).

        Returns:
            Tensor: SR frame: (b, 3, h, w).
        """
        if self.adapt_official_weights:
            lrs = lrs[:, [3, 0, 1, 2, 4, 5, 6], :, :, :]
        num_batches, num_lrs, _, h, w = lrs.size()
        lrs = self.normalize(lrs.view(-1, 3, h, w))
        lrs = lrs.view(num_batches, num_lrs, 3, h, w)
        lr_ref = lrs[:, self.ref_idx, :, :, :]
        lr_aligned = []
        for i in range(7):
            if i == self.ref_idx:
                lr_aligned.append(lr_ref)
            else:
                lr_supp = lrs[:, i, :, :, :]
                flow = self.spynet(lr_ref, lr_supp)
                lr_aligned.append(flow_warp(lr_supp, flow.permute(0, 2, 3, 1)))
        hr = torch.stack(lr_aligned, dim=1)
        hr = hr.view(num_batches, -1, h, w)
        hr = self.relu(self.conv_1(hr))
        hr = self.relu(self.conv_2(hr))
        hr = self.relu(self.conv_3(hr))
        hr = self.conv_4(hr) + lr_ref
        return self.denormalize(hr)


_reduction_modes = ['none', 'mean', 'sum']


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss
    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=weight[:, :, :-1, :])
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=weight[:, :, :, :-1])
        loss = x_diff + y_diff
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True, range_norm=False, perceptual_weight=1.0, style_weight=0.0, criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, range_norm=range_norm)
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:
            loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_ext.deform_conv_forward(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_ext.deform_conv_backward_input(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_ext.deform_conv_backward_parameters(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(f"convolution input is too small (output would be {'x'.join(map(str, output_size))})")
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, f'in_channels {in_channels} is not divisible by groups {groups}'
        assert out_channels % groups == 0, f'out_channels {out_channels} is not divisible by groups {groups}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        input_pad = x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1]
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()
        return out


class DeformConvPack(DeformConv):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ToyDiscriminator(nn.Module):

    def __init__(self):
        super(ToyDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(3, 4, 3, 1, 1, bias=True)
        self.bn0 = nn.BatchNorm2d(4, affine=True)
        self.conv1 = nn.Conv2d(4, 4, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(4, affine=True)
        self.linear = nn.Linear(4 * 6 * 6, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.bn0(self.conv0(x)))
        feat = self.lrelu(self.bn1(self.conv1(feat)))
        feat = feat.view(feat.size(0), -1)
        out = torch.sigmoid(self.linear(feat))
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (Blur,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EqualConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSDilationBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSRResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ModulatedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'num_style_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (NormStyleCode,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PredeblurModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RRDB,
     lambda: ([], {'num_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDBNet,
     lambda: ([], {'num_in_ch': 4, 'num_out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlockNoBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualDenseBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SFTUpBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPyNetTOF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 4, 64, 64])], {}),
     False),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TOFlow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 7, 3, 64, 64])], {}),
     False),
    (TSAFusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 4, 4])], {}),
     True),
    (ToRGB,
     lambda: ([], {'in_channels': 4, 'num_style_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (UpResBlock,
     lambda: ([], {'in_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xinntao_EDVR(_paritybench_base):
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

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

