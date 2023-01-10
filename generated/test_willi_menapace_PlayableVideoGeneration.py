import sys
_module = sys.modules[__name__]
del sys
build_evaluation_dataset = _module
dataset = _module
acquisition = _module
convert_annotated_video_directory = _module
convert_video_directory = _module
shift_video_ids = _module
split_and_resize_video = _module
subsample_videos_and_make_fixed_length = _module
train_val_test_split = _module
batching = _module
dataset_splitter = _module
transforms = _module
video = _module
video_dataset = _module
evaluate_dataset = _module
evaluation = _module
action_sampler = _module
action_variation_sampler = _module
dataset_evaluator = _module
dataset_evaluator_bair = _module
dataset_evaluator_breakout = _module
evaluation_dataset_builder = _module
evaluator = _module
metrics = _module
action_linear_classification = _module
action_variance = _module
breakout_platform_position = _module
detection_metric_1d = _module
detection_metric_2d = _module
fid = _module
fvd = _module
inception_score = _module
lpips = _module
motion_mask = _module
motion_masked_mse = _module
mse = _module
psnr = _module
ssim = _module
tennis_player_detector = _module
vgg_cosine_similarity = _module
density_plot = _module
density_plot_2d = _module
density_plot_2d_merged = _module
mean_vector_plot_2d = _module
results_file_plotter = _module
interpolate = _module
model = _module
layers = _module
centroid_estimator = _module
convolutional_lstm = _module
convolutional_lstm_cell = _module
final_block = _module
gumbel_softmax = _module
residual_block = _module
same_block = _module
up_block = _module
vgg = _module
main_model = _module
action_network = _module
conv_dynamics_network = _module
model = _module
rendering_network = _module
representation_network = _module
reduced_model = _module
action_network = _module
conv_dynamics_network = _module
model = _module
rendering_network = _module
representation_network = _module
play = _module
pytorch_fid = _module
fid_score = _module
inception = _module
train = _module
training = _module
losses = _module
smooth_mi_trainer = _module
trainer = _module
utils = _module
average_meter = _module
configuration = _module
dict_wrapper = _module
evaluation_configuration = _module
input_helper = _module
logger = _module
memory_displayer = _module
metrics_accumulator = _module
save_video_ffmpeg = _module
tensor_displayer = _module
tensor_folder = _module
tensor_resizer = _module
tensor_splitter = _module

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


import torchvision


import numpy as np


from typing import List


from typing import Tuple


from typing import Dict


import random


import torchvision.transforms as transforms


from typing import Set


import torchvision.transforms as tf


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


import matplotlib


import matplotlib.cm


from torchvision.utils import make_grid


import scipy


from scipy.stats import kurtosis


import torchvision.transforms as TF


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import tensorflow.compat.v1 as tf


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


import torch.utils.data


from torchvision.models.inception import inception_v3


from scipy.stats import entropy


import time


from torchvision import models


from functools import reduce


import collections


import math


from typing import Union


from sklearn.manifold import TSNE


import matplotlib.pyplot as plt


class BreakoutPlatformPosition(nn.Module):

    def __init__(self):
        super(BreakoutPlatformPosition, self).__init__()
        platform_color = [200, 72, 72]
        platform_color_lower = [100, 72, 72]
        self.lower_color_bound = torch.tensor(platform_color_lower, dtype=torch.float).unsqueeze(-1).unsqueeze(-1) / 255 - 0.15
        self.upper_color_bound = torch.tensor(platform_color, dtype=torch.float).unsqueeze(-1).unsqueeze(-1) / 255 + 0.15
        self.positions = None
        self.platform_row_scale = 188 / 208

    def create_positions_mask(self, height, width):
        """
        Creates a (height, width) tensor whose value in each point is the x coordinate

        :param height: The height of the tensor to create
        :param width: The width of the tensor to create
        :return:
        """
        mask = torch.arange(width).unsqueeze(0)
        mask = mask.repeat(height, 1)
        upper_limit = int(187 / 208 * height) + 1
        mask[:upper_limit] = 0
        self.platform_row = int(self.platform_row_scale * height)
        return mask

    def detect_platform(self, frame: np.ndarray) ->int:
        """
        Computes the position of the lower left part of the platform

        :param frame: (channels, height, width) boolean tensor with True in the positions of the frame where the
                                                platform color is detected
        :return: the x position of the left platform edge, -1 if none was found
        """
        width = frame.shape[-1]
        current_position_length = 0
        current_start_position = 0
        for idx in range(width):
            if frame[0, self.platform_row, idx] == True and idx != width - 1:
                if current_position_length == 0:
                    current_start_position = idx
                current_position_length += 1
            elif current_position_length > 0:
                if current_position_length > 11:
                    return current_start_position
                current_position_length = 0
        return -1

    def forward(self, observations: torch.Tensor) ->torch.Tensor:
        """
        Computes the position of the lower left part of the platform

        :param observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with x positions of the player-controlled bar
        """
        batch_size = observations.size(0)
        observations_count = observations.size(1)
        channels = observations.size(2)
        height = observations.size(3)
        width = observations.size(4)
        if self.positions is None:
            positions_mask = self.create_positions_mask(height, width)
            positions_mask = positions_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            positions_mask = positions_mask.repeat(1, observations_count, channels, 1, 1)
            self.positions = positions_mask
        current_positions_mask = self.positions.repeat(batch_size, 1, 1, 1, 1)
        platform_mask = torch.ge(observations, self.lower_color_bound) & torch.le(observations, self.upper_color_bound)
        platform_mask = platform_mask.cpu().numpy()
        all_positions = []
        for sequence_idx in range(batch_size):
            current_positions = []
            for observation_idx in range(observations_count):
                current_frame = platform_mask[sequence_idx, observation_idx]
                current_position = self.detect_platform(current_frame)
                current_positions.append(current_position)
            all_positions.append(current_positions)
        all_positions = np.asarray(all_positions)
        return all_positions


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
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


class FIDInceptionC(torchvision.models.inception.InceptionC):
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


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
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


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
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


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
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
            inception = _inception_v3(pretrained=True)
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

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class TensorFolder:

    @staticmethod
    def flatten(tensor: torch.Tensor) ->torch.Tensor:
        """
        Flattens the first two dimensions of the tensor

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1 * dim2, ...) tensor
        """
        tensor_size = list(tensor.size())
        flattened_tensor = torch.reshape(tensor, tuple([-1] + tensor_size[2:]))
        return flattened_tensor

    @staticmethod
    def flatten_list(tensors: List[torch.Tensor]) ->List[torch.Tensor]:
        """
        Applies flatten to all elements in the sequence
        See flatten for additional details
        """
        flattened_tensors = [TensorFolder.flatten(current_tensor) for current_tensor in tensors]
        return flattened_tensors

    @staticmethod
    def fold(tensor: torch.Tensor, second_dimension_size: torch.Tensor) ->torch.Tensor:
        """
        Separates the first tensor dimension into two separate dimensions of the given size

        :param tensor: (dim1 * second_dimension_size, ...) tensor
        :param second_dimension_size: the wished second dimension size for the output tensor
        :return: (dim1, second_dimension_size, ...) tensor
        """
        tensor_size = list(tensor.size())
        first_dimension_size = tensor_size[0]
        if first_dimension_size % second_dimension_size != 0:
            raise Exception(f'First dimension {first_dimension_size} is not a multiple of {second_dimension_size}')
        folded_first_dimension_size = first_dimension_size // second_dimension_size
        tensor = torch.reshape(tensor, [folded_first_dimension_size, second_dimension_size] + tensor_size[1:])
        return tensor

    @staticmethod
    def fold_list(tensors: List[torch.Tensor], second_dimension_size: torch.Tensor) ->List[torch.Tensor]:
        """
        Applies fold to each element in the sequence
        See fold for additional details
        """
        folded_tensors = [TensorFolder.fold(current_tensor, second_dimension_size) for current_tensor in tensors]
        return folded_tensors


class FID(nn.Module):

    def __init__(self):
        pass

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-06):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
            None
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_activation_statistics(self, dataloader, model):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(dataloader, model)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(self, dataloader, model):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : dataloader to use for image extraction
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                         Make sure that the number of samples is a multiple of
                         the batch size, otherwise some samples are ignored. This
                         behavior is retained to match the original FID score
                         implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()
        all_preds = []
        for current_batch in dataloader:
            batch_tuple = current_batch.to_tuple()
            observations, _, _, _ = batch_tuple
            observations = TensorFolder.flatten(observations)
            with torch.no_grad():
                pred = model(observations)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            all_preds.append(pred)
        return np.concatenate(all_preds, axis=0)

    def __call__(self, reference_dataloader, generated_dataloader) ->float:
        """
        Computes the FVD between the reference and the generated observations

        :param reference_dataloader: dataloader for reference observations
        :param generated_dataloader: dataloader for generated observations
        :return: The FVD between the two distributions
        """
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx])
        m1, s1 = self.calculate_activation_statistics(reference_dataloader, model)
        m2, s2 = self.calculate_activation_statistics(generated_dataloader, model)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return float(fid_value)


class LPIPS(nn.Module):

    def __init__(self):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net='vgg')

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) ->torch.Tensor:
        """
        Computes the psnr between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with psnr for each observation
        """
        observations_count = reference_observations.size(1)
        all_lpips = []
        for observation_idx in range(observations_count):
            current_reference_observations = reference_observations[:, observation_idx]
            current_generated_observations = generated_observations[:, observation_idx]
            lpips = self.metric(current_reference_observations, current_generated_observations, normalize=True)
            lpips = lpips.reshape(-1)
            all_lpips.append(lpips)
        return torch.stack(all_lpips, axis=1)


class MotionMaskCalculator:
    """
    Class for the creation of motion masks
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_frame_difference_motion_mask(observations):
        """
        :param observations: (bs, observations_count, 3, h, w) the observed sequences

        :return: (bs, observations_count, 1, h, w) tensor with the motion mask
        """
        sequence_length = observations.size(1)
        successor_observations = observations[:, 1:]
        predecessor_observations = observations[:, :-1]
        motion_mask = torch.abs(successor_observations - predecessor_observations)
        assert motion_mask.size(2) == 3
        motion_mask = motion_mask.sum(dim=2, keepdim=True) / 3
        motion_mask = torch.cat([torch.zeros_like(motion_mask[:, 0:1]), motion_mask], dim=1)
        return motion_mask


class MotionMaskedMSE(nn.Module):

    def __init__(self):
        super(MotionMaskedMSE, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) ->torch.Tensor:
        """
        Computes the mean squared error between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with MSE for each observation
        """
        motion_mask = MotionMaskCalculator.compute_frame_difference_motion_mask(reference_observations)
        differences = (reference_observations - generated_observations).pow(2) * motion_mask
        return torch.mean(differences, dim=[2, 3, 4])


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) ->torch.Tensor:
        """
        Computes the mean squared error between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with MSE for each observation
        """
        return torch.mean((reference_observations - generated_observations).pow(2), dim=[2, 3, 4])


class PSNR(nn.Module):

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) ->torch.Tensor:
        """
        Computes the psnr between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with psnr for each observation
        """
        EPS = 1e-08
        reference_observations = reference_observations / range
        generated_observations = generated_observations / range
        mse = torch.mean((reference_observations - generated_observations) ** 2, dim=[2, 3, 4])
        score = -10 * torch.log10(mse + EPS)
        return score


class SSIM(nn.Module):

    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) ->torch.Tensor:
        """
        Computes the ssim between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with ssim for each observation
        """
        observations_count = reference_observations.size(1)
        flattened_reference_observations = TensorFolder.flatten(reference_observations)
        flattened_generated_observations = TensorFolder.flatten(generated_observations)
        flattened_ssim = ssim(flattened_generated_observations, flattened_reference_observations, range, reduction='none')
        folded_ssim = TensorFolder.fold(flattened_ssim, observations_count)
        return folded_ssim


class TennisPlayerDetector(nn.Module):

    def __init__(self):
        super(TennisPlayerDetector, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.threshold = 0.8
        self.COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def check_box_boundaries(self, box):
        if box[2] <= 60 and box[1] <= 26:
            return False
        if box[0] >= 200 and box[1] <= 26:
            return False
        if box[1] > 80:
            return False
        return True

    def compute_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def forward(self, observations):
        """
        Computes the mean squared error between the reference and the generated observations

        :param observations: (bs, observations_count, channels, height, width) tensor with observations
        :return: (bs, observations_count, 2) tensor with x and y coordinates of the detection, -1 if any
        """
        batch_size = observations.size(0)
        observations_count = observations.size(1)
        all_predicted_centers = []
        for observations_idx in range(observations_count):
            current_observations = observations[:, observations_idx]
            with torch.no_grad():
                predictions = self.model(current_observations)
            for current_prediction in predictions:
                pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(current_prediction['labels'].cpu().numpy())]
                pred_boxes = [(i[0], i[1], i[2], i[3]) for i in list(current_prediction['boxes'].detach().cpu().numpy())]
                pred_score = list(current_prediction['scores'].detach().cpu().numpy())
                filtered_preds = [pred_score.index(x) for x in pred_score if x > self.threshold]
                if len(filtered_preds) > 0:
                    pred_t = filtered_preds[-1]
                    pred_boxes = pred_boxes[:pred_t + 1]
                    pred_class = pred_class[:pred_t + 1]
                else:
                    pred_boxes = []
                    pred_class = []
                matches = []
                for idx in range(len(pred_boxes)):
                    if pred_class[idx] == 'person':
                        if self.check_box_boundaries(pred_boxes[idx]):
                            matches.append((pred_boxes[idx][3] - pred_boxes[idx][1], pred_boxes[idx]))
                if len(matches) == 0:
                    all_predicted_centers.append([-1, -1])
                else:
                    if len(matches) > 1:
                        None
                    matches.sort(key=lambda x: x[0])
                    all_predicted_centers.append(self.compute_center(matches[-1][-1]))
        predicted_centers = np.asarray(all_predicted_centers).reshape((observations_count, batch_size, 2))
        return np.moveaxis(predicted_centers, 0, 1)


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) ->List[torch.Tensor]:
        """

        :param x: (bs, 3, height, width) tensor representing the input image
        :return: List of (bs, features_i, height_i, width_i) tensors representing vgg features at different levels
        """
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGCosineSimilarity(nn.Module):

    def __init__(self):
        super(VGGCosineSimilarity, self).__init__()
        self.vgg = Vgg19()
        self.vgg = self.vgg

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) ->torch.Tensor:
        """
        Computes the VGG Cosine Similarity between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with ssim for each observation
        """
        reference_observations = reference_observations / range
        generated_observations = generated_observations / range
        normalization_mean = 0.5
        normaliation_std = 0.5
        normalization_eps = 1e-06
        reference_observations = (reference_observations - normalization_mean) / (normaliation_std + normalization_eps)
        generated_observations = (generated_observations - normalization_mean) / (normaliation_std + normalization_eps)
        bs = reference_observations.size(0)
        observations_count = reference_observations.size(1)
        flattened_reference_observations = TensorFolder.flatten(reference_observations)
        flattened_generated_observations = TensorFolder.flatten(generated_observations)
        flattened_reference_features = self.vgg(flattened_reference_observations)
        flattened_generated_observations = self.vgg(flattened_generated_observations)
        features_count = len(flattened_reference_features)
        similarities = torch.zeros((bs, observations_count), dtype=torch.float)
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-06)
        for current_reference_feature, current_generated_feature in zip(flattened_reference_features, flattened_generated_observations):
            current_reference_feature = current_reference_feature.reshape((bs * observations_count, -1))
            current_generated_feature = current_generated_feature.reshape((bs * observations_count, -1))
            similarities += TensorFolder.fold(cosine_similarity(current_reference_feature, current_generated_feature), observations_count)
        similarities /= features_count
        return similarities


class CentroidEstimator(nn.Module):
    """
    Estimator for centroid positions
    """

    def __init__(self, centroids_count: int, space_dimensions: int, alpha: float):
        """
        Initializes the model

        :param centroids_count: the number of centroids to track
        :param space_dimensions: dimension of the space where the centroids lie
        :param alpha: value to use for the moving average computation
        """
        super(CentroidEstimator, self).__init__()
        self.centroids_count = centroids_count
        self.alpha = alpha
        self.space_dimensions = space_dimensions
        initial_centroids = torch.randn((self.centroids_count, self.space_dimensions), dtype=torch.float32)
        self.estimated_centroids = nn.Parameter(initial_centroids, requires_grad=False)

    def get_estimated_centroids(self) ->torch.tensor:
        """
        Obtains the estimates for the centroids
        :return: (centroids_count, space_dimensions) tensor with estimated centroids
        """
        return self.estimated_centroids

    def update_centroids(self, points_priors: torch.Tensor, centroid_assignments: torch.Tensor):
        """

        :param points_priors: (..., 2, space_dimensions) tensor with (mean, variance) for each point
        :param centroid_assignments: (..., centroids_count) tensor with cluster assignment probabilities in 0, 1
                                                            for each point
        :return:
        """
        if not self.training:
            return
        points_priors = points_priors.view((-1, 2, self.space_dimensions))
        point_means = points_priors[:, 0]
        centroid_assignments = centroid_assignments.view((-1, self.centroids_count))
        point_means = point_means.unsqueeze(1)
        unsqueezed_centroid_assignments = centroid_assignments.unsqueeze(-1)
        current_centroid_estimate = (point_means * unsqueezed_centroid_assignments).sum(0)
        mean_weights = centroid_assignments.sum(0).unsqueeze(-1)
        current_centroid_estimate = current_centroid_estimate / mean_weights
        return_centroids = self.estimated_centroids * (1 - self.alpha) + current_centroid_estimate * self.alpha
        self.estimated_centroids.data = return_centroids.detach()

    def compute_variations(self, points: torch.Tensor, centroid_assignments: torch.Tensor):
        """
        Compute the variation vector of points with respect to centroids

        :param points: (..., space_dimensions) tensor with each point
        :param centroid_assignments: (..., centroids_count) tensor with cluster assignment probabilities in 0, 1
                                                            for each point
        :return: (..., space_dimensions) tensor with variation of each point with respect to centroids
        """
        initial_dimensions = list(points.size())[:-1]
        points = points.view((-1, self.space_dimensions))
        centroid_assignments = centroid_assignments.view((-1, self.centroids_count))
        variations = points.unsqueeze(1) - self.estimated_centroids
        variations = centroid_assignments.unsqueeze(-1) * variations
        variations = variations.sum(1)
        variations = variations.reshape(tuple(initial_dimensions + [-1]))
        return variations


class ConvLSTMCell(nn.Module):
    """
    A Convolutional LSTM Cell
    """

    def __init__(self, in_planes: int, out_planes: int):
        """

        :param in_planes: Number of input channels
        :param out_planes: Number of output channels
        """
        super(ConvLSTMCell, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.input_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.forget_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.cell_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def make_2d_tensor(self, tensor: torch.Tensor, height: int, width: int) ->torch.Tensor:
        """
        Transforms a 1d tensor into a 2d tensor of specified dimensions

        :param tensor: (bs, features) tensor
        :return: (bs, features, height, width) tensor with repeated features along the spatial dimensions
        """
        tensor = tensor.unsqueeze(dim=-1).unsqueeze(dim=-1)
        tensor = tensor.repeat((1, 1, height, width))
        return tensor

    def channelwise_concat(self, inputs: List[torch.Tensor]):
        """
        Concatenates all inputs tensors channelwise

        :param inputs: [(bs, features_i, height, width) \\ (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :return:
        """
        height = 0
        width = 0
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                height = current_tensor.size(2)
                width = current_tensor.size(3)
                break
        if height == 0 or width == 0:
            raise Exception('No tensor in the inputs has a spatial dimension. Ensure at least one tensor represents a tensor with spatial dimensions')
        expanded_tensors = []
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                expanded_tensors.append(current_tensor)
            elif len(current_tensor.size()) == 2:
                expanded_tensors.append(self.make_2d_tensor(current_tensor, height, width))
            else:
                raise Exception('Expected tensors with 2 or 4 dimensions')
        concatenated_tensor = torch.cat(expanded_tensors, dim=1)
        total_features = concatenated_tensor.size(1)
        if total_features != self.in_planes + self.out_planes:
            raise Exception(f'The input tensors features sum to {total_features}, but layer takes {self.in_planes} features as input')
        return concatenated_tensor

    def forward(self, inputs: List[torch.Tensor], hidden_states: torch.Tensor, hidden_cell_states: torch.Tensor) ->torch.Tensor:
        """
        Computes the successor states given the inputs

        :param inputs: [(bs, features_i, height, width) \\ (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :param hidden_states: (bs, out_planes, height, width) tensor with hidden state
        :param hidden_cell_states: (bs, out_planes, height, width) tensor with hidden cell state

        :return: (bs, out_planes, height, width), (bs, out_planes, height, width) tensors with hidden_state and hidden_cell_state
        """
        inputs.append(hidden_states)
        concatenated_input = self.channelwise_concat(inputs)
        i = torch.sigmoid(self.input_gate(concatenated_input))
        f = torch.sigmoid(self.forget_gate(concatenated_input))
        o = torch.sigmoid(self.output_gate(concatenated_input))
        c = torch.tanh(self.cell_gate(concatenated_input))
        successor_hidden_cell_states = f * hidden_cell_states + i * c
        successor_hidden_state = o * torch.tanh(successor_hidden_cell_states)
        return successor_hidden_state, successor_hidden_cell_states


class ConvLSTM(nn.Module):
    """
    A Convolutional LSTM
    """

    def __init__(self, in_planes: int, out_planes: int, size: Tuple[int]):
        """

        :param in_planes: Number of input channels
        :param out_planes: Number of output channels
        :param size: (height, width) of the input tensors
        """
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(in_planes, out_planes)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.height = size[0]
        self.width = size[1]
        self.initial_hidden_state = nn.Parameter(torch.zeros(self.out_planes, self.height, self.width))
        self.initial_hidden_cell_state = nn.Parameter(torch.zeros(self.out_planes, self.height, self.width))

    def reinit_memory(self, batch_size: int):
        """
        Initializes the cell state
        :param batch_size: Batch size of all the successive inputs until the next reinit_memory call
        :return:
        """
        if hasattr(self, 'current_hidden_state'):
            del self.current_hidden_state
            del self.current_hidden_cell_state

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        """
        Computes the successor states given the current inputs
        Current states are maintained implicitly and are reset through reinit_memory

        reinit_memory must have been called at least once before forward

        :param inputs: [(bs, features_i, height, width) \\ (bs, features_i)] list of tensor which feature dimensions sum to in_planes

        :return: (bs, out_planes, height, width) tensor with the successor states
        """
        batch_size = inputs[0].size(0)
        if not hasattr(self, 'current_hidden_state'):
            self.current_hidden_state = self.initial_hidden_state.repeat((batch_size, 1, 1, 1))
            self.current_hidden_cell_state = self.initial_hidden_cell_state.repeat((batch_size, 1, 1, 1))
        cell_output = self.cell(inputs, self.current_hidden_state, self.current_hidden_cell_state)
        self.current_hidden_state, self.current_hidden_cell_state = cell_output
        return self.current_hidden_state


class FinalBlock(nn.Module):
    """
    Final block transforming features into an image
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        """

        :param in_features: Input features to the module
        :param out_features: Output feature
        """
        super(FinalBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x


class GumbelSoftmax(nn.Module):
    """
    Module for gumbel sampling
    """

    def __init__(self, initial_temperature, hard=True):
        """
        Initializes the samples to operate at the given temperature
        :param initial_temperature: initial temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :param hard: if true uses the hard straight through gumbel implementation
        """
        super(GumbelSoftmax, self).__init__()
        self.current_temperature = initial_temperature
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-20):
        """
        Samples gumbel variable with given shape
        :param shape: shape of the variable to output
        :param eps: constant for numeric stability
        :return: (*shape) tensor with gumbel samples
        """
        U = torch.rand(shape)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_soft_sample(self, input):
        """
        Computes soft gumbel samples

        :param input: (bs, classes_count) tensor representing log of probabilities
        :return: (bs, classes_count) soft samples
        """
        y = input + self.sample_gumbel(input.size())
        return F.softmax(y / self.current_temperature, dim=-1)

    def forward(self, input, temperature=None):
        """

        :param input: (bs, classes_count) tensor representing log of probabilities
        :param temperature: new temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :return:
        """
        if temperature is not None:
            self.current_temperature = temperature
        soft_samples = self.gumbel_soft_sample(input)
        if self.hard:
            shape = soft_samples.size()
            _, ind = soft_samples.max(dim=-1)
            y_hard = torch.zeros_like(soft_samples).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            hard_samples = (y_hard - soft_samples).detach() + soft_samples
            return hard_samples
        return soft_samples


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Residual block
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, downsample_factor=1, last_affine=True, drop_final_activation=False):
        """

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        """
        super(ResidualBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes, affine=last_affine)
        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation
        self.downsample = None
        if self.downsample_factor != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(conv1x1(in_planes, out_planes, stride=1), nn.AvgPool2d(downsample_factor), norm_layer(out_planes, affine=last_affine))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if not self.drop_final_activation:
            out = self.relu(out)
        return out


class SameBlock(nn.Module):
    """
    Convolutional block with normalization and activation
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, downsample_factor=1, drop_final_activation=False):
        """

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        """
        super(SameBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation
        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        if not self.drop_final_activation:
            out = self.relu(out)
        return out


class UpBlock(nn.Module):
    """
    Upsampling block.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, scale_factor=2, upscaling_mode='nearest', late_upscaling=False):
        """

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param kernel_size: Size of the kernel
        :param padding: Size of padding
        :param scale_factor: Multiplicative factor such that output_res = input_res * scale_factor
        :param upscaling_mode: interpolation upscaling mode
        :param late_upscaling: if True upscaling is applied at the end of the block, otherwise it is applied at the beginning
        """
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor
        self.upscaling_mode = upscaling_mode
        self.late_upscaling = late_upscaling
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = x
        if not self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)
        out = self.conv(out)
        out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)
        return out


class TensorSplitter:

    @staticmethod
    def predecessor_successor_split(tensor: torch.Tensor) ->torch.Tensor:
        """
        Splits a tensor into the second dimension predecessors and successors

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1, 0:dim2-1, ...), (dim1, 1:dim2, ...) tensor
        """
        predecessor_tensor = tensor[:, :-1]
        successor_tensor = tensor[:, 1:]
        return predecessor_tensor, successor_tensor


class ActionNetwork(nn.Module):
    """
    Model that reconstructs the frame associated to a state
    """

    def __init__(self, config):
        super(ActionNetwork, self).__init__()
        self.config = config
        self.state_features = config['model']['representation_network']['state_features']
        self.actions_count = config['data']['actions_count']
        self.action_space_dimension = config['model']['action_network']['action_space_dimension']
        residual_blocks = [ResidualBlock(self.state_features, 2 * self.state_features, downsample_factor=2), ResidualBlock(2 * self.state_features, 2 * self.state_features, downsample_factor=1)]
        self.residuals = nn.Sequential(*residual_blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mean_fc = nn.Linear(2 * self.state_features, self.action_space_dimension)
        self.variance_fc = nn.Linear(2 * self.state_features, self.action_space_dimension)
        self.final_fc = nn.Linear(self.action_space_dimension, self.actions_count)

    def sample(self, mean: torch.Tensor, variance: torch.Tensor):
        """
        Samples from the posterior distribution with given mean and variance

        :param mean: (..., action_space_dimension) tensor with posterior mean
        :param variance: (..., action_space_dimension) tensor with posterior variance
        :return: (..., action_space_dimension) tensor with points sampled from the posterior
        """
        noise = torch.randn(mean.size(), dtype=torch.float32)
        sampled_points = noise * torch.sqrt(variance) + mean
        return sampled_points

    def split_batch(self, tensor: torch.Tensor):
        """
        Splits a tensor in half following the first dimension
        Tensor must have an even number of elements in the first dimension
        :param tensor: (bs, ...) tensor to split
        :return: (bs/2, ...), (bs/2, ...) split tensors
        """
        batch_size = tensor.size(0)
        assert batch_size % 2 == 0
        return tensor[:batch_size], tensor[batch_size:]

    def forward(self, states: torch.Tensor, states_attention: torch.Tensor) ->torch.Tensor:
        """
        Computes actions corresponding to the state transition from predecessor to successor state

        :param states: (bs, observations_count, states_features, states_height, states_width) tensor
        :param states_attention: (bs, observations_count, 1, states_height, states_width) tensor with attention

        :return: action_probabilities, action_directions_distribution, sampled_action_directions,
                 action_states_distribution, sampled_action_states
                 (bs, observations_count - 1, actions_count) tensor with logits of probabilities for each action
                 (bs, observations_count - 1, 2, action_space_dimension) tensor posterior mean and variance for action directions
                 (bs, observations_count - 1, action_space_dimension) tensor with sampled action directions
                 (bs, observations_count, 2, action_space_dimension) tensor posterior mean and variance for action states
                 (bs, observations_count, action_space_dimension) tensor with sampled action states

        """
        attentive_states = states * states_attention
        observations_count = attentive_states.size(1)
        flat_attentive_states = TensorFolder.flatten(attentive_states)
        x = self.residuals(flat_attentive_states)
        x = self.gap(x)
        flat_states_mean = self.mean_fc(x.view(x.size(0), -1))
        flat_states_variance = torch.abs(self.variance_fc(x.view(x.size(0), -1)))
        flat_states_distribution = torch.stack([flat_states_mean, flat_states_variance], dim=1)
        flat_sampled_states = self.sample(flat_states_mean, flat_states_variance)
        folded_states_mean = TensorFolder.fold(flat_states_mean, observations_count)
        folded_states_variance = TensorFolder.fold(flat_states_variance, observations_count)
        folded_states_distribution = TensorFolder.fold(flat_states_distribution, observations_count)
        folded_sampled_states = TensorFolder.fold(flat_sampled_states, observations_count)
        predecessor_mean, successor_mean = TensorSplitter.predecessor_successor_split(folded_states_mean)
        predecessor_variance, successor_variance = TensorSplitter.predecessor_successor_split(folded_states_variance)
        action_directions_mean = successor_mean - predecessor_mean
        action_directions_variance = successor_variance + predecessor_variance
        action_directions_distribution = torch.stack([action_directions_mean, action_directions_variance], dim=2)
        sampled_action_directions = self.sample(action_directions_mean, action_directions_variance)
        flat_sampled_action_directions = TensorFolder.flatten(sampled_action_directions)
        flat_action_probabilities = self.final_fc(flat_sampled_action_directions)
        folded_action_probabilities = TensorFolder.fold(flat_action_probabilities, observations_count - 1)
        return folded_action_probabilities, action_directions_distribution, sampled_action_directions, folded_states_distribution, folded_sampled_states


class ConvDynamicsNetwork(nn.Module):
    """
    Model that predicts the future state given the current state and an action
    """

    def __init__(self, config):
        super(ConvDynamicsNetwork, self).__init__()
        self.hidden_state_size = config['model']['dynamics_network']['hidden_state_size']
        self.random_noise_size = config['model']['dynamics_network']['random_noise_size']
        self.state_resolution = config['model']['representation_network']['state_resolution']
        self.state_features = config['model']['representation_network']['state_features']
        actions_count = config['data']['actions_count']
        actions_space_dimension = config['model']['action_network']['action_space_dimension']
        auxiliary_input_size = actions_count + actions_space_dimension
        self.recurrent_layers = [ConvLSTM(self.state_features + auxiliary_input_size, self.hidden_state_size, self.state_resolution), ConvLSTM(2 * self.hidden_state_size + auxiliary_input_size, 2 * self.hidden_state_size, (self.state_resolution[0] // 2, self.state_resolution[1] // 2)), ConvLSTM(self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, self.state_resolution)]
        self.recurrent_layers_blocks = nn.ModuleList([nn.Sequential(self.recurrent_layers[0], nn.BatchNorm2d(self.hidden_state_size)), nn.Sequential(self.recurrent_layers[1], nn.BatchNorm2d(2 * self.hidden_state_size)), nn.Sequential(self.recurrent_layers[2], nn.BatchNorm2d(self.hidden_state_size))])
        self.non_recurrent_blocks = nn.ModuleList([SameBlock(self.hidden_state_size + auxiliary_input_size, 2 * self.hidden_state_size, downsample_factor=2), UpBlock(2 * self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, upscaling_mode='bilinear', late_upscaling=True), SameBlock(self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, downsample_factor=1)])

    def reinit_memory(self, batch_size: int):
        """
        Initializes the state of the recurrent cells
        :param batch_size: Batch size of all the successive inputs until the next reinit_memory call
        :return:
        """
        for current_layer in self.recurrent_layers:
            current_layer.reinit_memory(batch_size)

    def make_2d_tensor(self, tensor: torch.Tensor, height: int, width: int) ->torch.Tensor:
        """
        Transforms a 1d tensor into a 2d tensor of specified dimensions

        :param tensor: (bs, features) tensor
        :return: (bs, features, height, width) tensor with repeated features along the spatial dimensions
        """
        tensor = tensor.unsqueeze(dim=-1).unsqueeze(dim=-1)
        tensor = tensor.repeat((1, 1, height, width))
        return tensor

    def channelwise_concat(self, inputs: List[torch.Tensor]):
        """
        Concatenates all inputs tensors channelwise

        :param inputs: [(bs, features_i, height, width) \\ (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :return:
        """
        height = 0
        width = 0
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                height = current_tensor.size(2)
                width = current_tensor.size(3)
                break
        if height == 0 or width == 0:
            raise Exception('No tensor in the inputs has a spatial dimension. Ensure at least one tensor represents a tensor with spatial dimensions')
        expanded_tensors = []
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                expanded_tensors.append(current_tensor)
            elif len(current_tensor.size()) == 2:
                expanded_tensors.append(self.make_2d_tensor(current_tensor, height, width))
            else:
                raise Exception('Expected tensors with 2 or 4 dimensions')
        concatenated_tensor = torch.cat(expanded_tensors, dim=1)
        return concatenated_tensor

    def forward(self, states: torch.Tensor, actions: torch.Tensor, variations: torch.Tensor, random_noise: torch.Tensor) ->torch.Tensor:
        """
        Computes the successor states given the selected actions and noise
        Current states are maintained implicitly and are reset through reinit_memory
        reinit_memory must have been called at least once before forward

        :param states: (bs, states_features, states_height, states_width) tensor
        :param actions: (bs, actions_count) tensor with actions probabilities
        :param variations: (bs, action_space_dimension) tensor with action variations
        :param random_noise: (bs, random_noise_size, states_height, states_width) tensor with random noise

        :return: (bs, hidden_state_size) tensor with the successor states
        """
        states = self.recurrent_layers_blocks[0]([states, actions, variations])
        states = self.non_recurrent_blocks[0](self.channelwise_concat([states, actions, variations]))
        states = self.recurrent_layers_blocks[1]([states, actions, variations])
        states = self.non_recurrent_blocks[1](self.channelwise_concat([states, actions, variations]))
        states = self.recurrent_layers_blocks[2]([states, actions, variations])
        states = self.non_recurrent_blocks[2](self.channelwise_concat([states, actions, variations]))
        return states


class RenderingNetwork(nn.Module):
    """
    Model that reconstructs the frame associated to a hidden state
    """

    def __init__(self, config):
        super(RenderingNetwork, self).__init__()
        self.config = config
        self.hidden_state_size = config['model']['dynamics_network']['hidden_state_size']
        bottleneck_block_list = []
        upsample_block_list = [nn.Sequential(UpBlock(64, 64, scale_factor=2, upscaling_mode='bilinear'), ResidualBlock(64, 64, downsample_factor=1)), nn.Sequential(UpBlock(64, 32, scale_factor=2, upscaling_mode='bilinear'), ResidualBlock(32, 32, downsample_factor=1)), UpBlock(32, 16, scale_factor=2, upscaling_mode='bilinear')]
        final_block_list = [FinalBlock(64, 3, kernel_size=3, padding=1), FinalBlock(32, 3, kernel_size=3, padding=1), FinalBlock(16, 3, kernel_size=7, padding=3)]
        self.bottleneck_blocks = nn.Sequential(*bottleneck_block_list)
        self.upsample_blocks = nn.ModuleList(upsample_block_list)
        self.final_blocks = nn.ModuleList(final_block_list)
        if len(upsample_block_list) != len(final_block_list):
            raise Exception('Rendering network specifies a number of upsampling blocks that differs from the number of final blocks')

    def forward(self, hidden_states: torch.Tensor) ->torch.Tensor:
        """
        Computes the frames corresponding to each state at multiple resolutions

        :param hidden_states: (bs, hidden_state_size, state) tensor
        :return: (bs, 3, height, width), [(bs, 3, height/2^i, width/2^i) for i in range(num_upsample_blocks)]
        """
        current_features = self.bottleneck_blocks(hidden_states)
        reconstructed_observations = []
        for upsample_block, final_block in zip(self.upsample_blocks, self.final_blocks):
            current_features = upsample_block(current_features)
            current_reconstructed_observation = final_block(current_features)
            reconstructed_observations.append(current_reconstructed_observation)
        reconstructed_observations = list(reversed(reconstructed_observations))
        return reconstructed_observations[0], reconstructed_observations


class RepresentationNetwork(nn.Module):
    """
    Model that encodes an observation into a state with action attention
    """

    def __init__(self, config):
        super(RepresentationNetwork, self).__init__()
        self.config = config
        self.in_features = self.config['training']['batching']['observation_stacking'] * 3
        self.conv1 = nn.Conv2d(self.in_features, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        residual_blocks = [ResidualBlock(16, 16, downsample_factor=1), ResidualBlock(16, 32, downsample_factor=2), ResidualBlock(32, 32, downsample_factor=1), ResidualBlock(32, 64, downsample_factor=2), ResidualBlock(64, 64, downsample_factor=1), ResidualBlock(64, 64 + 1, downsample_factor=1)]
        self.residuals = nn.Sequential(*residual_blocks)

    def forward(self, observations: torch.Tensor) ->torch.Tensor:
        """
        Computes the state corresponding to each observation

        :param observations: (bs, 3 * observation_stacking, height, width) tensor
        :return: (bs, states_features, states_height, states_width) tensor of states
                 (bs, 1, states_height, states_width) tensor with attention
        """
        x = self.conv1(observations)
        x = F.avg_pool2d(x, 2)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.residuals(x)
        state = x[:, :-1]
        attention = x[:, -1:]
        attention_shape = attention.size()
        attention_flat_shape = [attention_shape[0], attention_shape[1], attention_shape[2] * attention_shape[3]]
        flat_attention = attention.reshape(attention_flat_shape)
        flat_attention = torch.sigmoid(flat_attention)
        attention = flat_attention.reshape(attention_shape)
        return state, attention


def model(config):
    return Model(config)


class FixedMatrixEstimator(nn.Module):

    def __init__(self, rows, columns, initial_alpha=0.2, initial_value=None):
        """
        Initializes the joint probability estimator for a (rows, columns) matrix with the given fixed alpha factor
        :param rows, columns: Dimension of the probability matrix to estimate
        :param initial_alpha: Value to use assign for alpha
        """
        super(FixedMatrixEstimator, self).__init__()
        self.alpha = initial_alpha
        if initial_value is None:
            initial_value = torch.tensor([[1.0 / (rows * columns)] * columns] * rows, dtype=torch.float32)
        self.estimated_matrix = nn.Parameter(initial_value, requires_grad=False)

    def forward(self, latest_probability_matrix):
        return_matrix = self.estimated_matrix * (1 - self.alpha) + latest_probability_matrix * self.alpha
        self.estimated_matrix.data = return_matrix.detach()
        return return_matrix


class MutualInformationLoss(nn.Module):

    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor, distribution_2: torch.Tensor) ->torch.Tensor:
        """
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        """
        dim = distribution_1.size(-1)
        assert distribution_2.size(-1) == dim
        distribution_1 = distribution_1.view(-1, dim)
        distribution_2 = distribution_2.view(-1, dim)
        batch_size = distribution_1.size(0)
        assert distribution_2.size(0) == batch_size
        p_i_j = distribution_1.unsqueeze(2) * distribution_2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.0
        p_i_j = p_i_j / p_i_j.sum()
        return p_i_j

    def __call__(self, distribution_1: torch.Tensor, distribution_2: torch.Tensor, lamb=1.0, eps=sys.float_info.epsilon) ->torch.Tensor:
        """
        Computes the mutual information loss for a joint probability matrix
        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :param lamb: lambda parameter to change the importance of entropy in the loss
        :param eps: small constant for numerical stability
        :return: mutual information loss for the given joint probability matrix
        """
        joint_probability_matrix = self.compute_joint_probability_matrix(distribution_1, distribution_2)
        rows, columns = joint_probability_matrix.size()
        marginal_rows = joint_probability_matrix.sum(dim=1).view(rows, 1).expand(rows, columns)
        marginal_columns = joint_probability_matrix.sum(dim=0).view(1, columns).expand(rows, columns)
        joint_probability_matrix[(joint_probability_matrix < eps).data] = eps
        marginal_rows = marginal_rows.clone()
        marginal_columns = marginal_columns.clone()
        marginal_rows[(marginal_rows < eps).data] = eps
        marginal_columns[(marginal_columns < eps).data] = eps
        mutual_information = joint_probability_matrix * (torch.log(joint_probability_matrix) - lamb * torch.log(marginal_rows) - lamb * torch.log(marginal_columns))
        mutual_information = mutual_information.sum()
        return -1 * mutual_information


class SmoothMutualInformationLoss(MutualInformationLoss):
    """
    Mutual information loss with smooth joint probability matrix estimation
    """

    def __init__(self, config):
        """
        Creates the loss according to the specified configuration
        :param config: The configuration
        """
        super(SmoothMutualInformationLoss, self).__init__()
        self.actions_count = config['data']['actions_count']
        self.mi_estimation_alpha = config['training']['mutual_information_estimation_alpha']
        self.matrix_estimator = FixedMatrixEstimator(self.actions_count, self.actions_count, self.mi_estimation_alpha)

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor, distribution_2: torch.Tensor) ->torch.Tensor:
        """
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        """
        current_joint_probability_matrix = super(SmoothMutualInformationLoss, self).compute_joint_probability_matrix(distribution_1, distribution_2)
        smoothed_joint_probability_matrix = self.matrix_estimator(current_joint_probability_matrix)
        return smoothed_joint_probability_matrix


class UnmeanedPerceptualLoss(nn.Module):

    def __init__(self):
        super(UnmeanedPerceptualLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg = self.vgg

    def forward(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None) ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes the perceptual loss between the sets of observations

        :param observations: (bs, observations_count, 3*observation_stacking, h, w) ground truth observations. Rescaled if needed
        :param reconstructed_observations: (bs, observations_count|observations_count-1, 3, height, width) tensor with reconstructed observations
        :param weight_mask: (bs, observations_count, 1, h, w) tensor weights to assign to each spatial position for loss computation. Rescaled if needed

        :return: total_loss, individual_losses Perceptual loss between ground truth and reconstructed observations. Both the total loss and the loss
                 for each vgg feature level are returned. Losses have a batch size dimension
        """
        ground_truth_observations = observations[:, :, :3]
        sequence_length = ground_truth_observations.size(1)
        reconstructed_sequence_length = reconstructed_observations.size(1)
        original_observation_height = ground_truth_observations.size(3)
        original_observation_width = ground_truth_observations.size(4)
        height = reconstructed_observations.size(3)
        width = reconstructed_observations.size(4)
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(f'Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}')
            ground_truth_observations = ground_truth_observations[:, 1:]
        if weight_mask is not None:
            weight_mask_length = weight_mask.size(1)
            if weight_mask_length != reconstructed_sequence_length:
                if reconstructed_sequence_length != weight_mask_length - 1:
                    raise Exception(f'Received a reconstructed sequence with length {reconstructed_sequence_length}, but got a weight mast of length {weight_mask_length}')
                weight_mask = weight_mask[:, 1:]
            weight_height = weight_mask.size(3)
            weight_width = weight_mask.size(4)
            flat_weight_shape = -1, 1, weight_height, weight_width
            flattened_weight_mask = weight_mask.reshape(flat_weight_shape)
        flattened_ground_truth_observations = TensorFolder.flatten(ground_truth_observations)
        flattened_reconstructed_observations = TensorFolder.flatten(reconstructed_observations)
        if original_observation_width != width or original_observation_height != height:
            flattened_ground_truth_observations = F.interpolate(flattened_ground_truth_observations, (height, width), mode='bilinear')
        with torch.no_grad():
            ground_truth_vgg_features = self.vgg(flattened_ground_truth_observations.detach())
        reconstructed_vgg_features = self.vgg(flattened_reconstructed_observations)
        total_loss = None
        single_losses = []
        for current_ground_truth_feature, current_reconstructed_feature in zip(ground_truth_vgg_features, reconstructed_vgg_features):
            if weight_mask is None:
                current_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature).mean(dim=[1, 2, 3])
            else:
                current_feature_channels = current_ground_truth_feature.size(1)
                current_feature_height = current_ground_truth_feature.size(2)
                current_feature_width = current_ground_truth_feature.size(3)
                scaled_weight_masks = F.interpolate(flattened_weight_mask, size=(current_feature_height, current_feature_width), mode='bilinear', align_corners=False)
                unreduced_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature)
                unreduced_loss = unreduced_loss * scaled_weight_masks
                current_loss = unreduced_loss.sum(dim=(1, 2, 3))
                current_loss = current_loss / (scaled_weight_masks.sum(dim=(1, 2, 3)) * current_feature_channels)
            if total_loss is None:
                total_loss = current_loss
            else:
                total_loss += current_loss
            single_losses.append(current_loss)
        return total_loss, single_losses


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BreakoutPlatformPosition,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 3, 4, 4])], {}),
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
    (FinalBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FixedMatrixEstimator,
     lambda: ([], {'rows': 4, 'columns': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GumbelSoftmax,
     lambda: ([], {'initial_temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSNR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SameBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpBlock,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_willi_menapace_PlayableVideoGeneration(_paritybench_base):
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

