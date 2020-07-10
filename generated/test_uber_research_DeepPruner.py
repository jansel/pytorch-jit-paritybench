import sys
_module = sys.modules[__name__]
del sys
demo_script = _module
models = _module
config = _module
feature_extractor = _module
image_reconstruction = _module
patch_match = _module
dataloader = _module
kitti_collector = _module
kitti_loader = _module
kitti_submission_collector = _module
preprocess = _module
readpfm = _module
sceneflow_collector = _module
sceneflow_loader = _module
finetune_kitti = _module
loss_evaluation = _module
deeppruner = _module
feature_extractor_best = _module
feature_extractor_fast = _module
patch_match = _module
submodules = _module
submodules2d = _module
submodules3d = _module
utils = _module
setup_logging = _module
submission_kitti = _module
train_sceneflow = _module

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
xrange = range
wraps = functools.wraps


import torch


import random


import numpy as np


import torchvision.transforms as transforms


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


import logging


import math


from collections import namedtuple


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


from torchvision import transforms


import time


class feature_extractor(nn.Module):

    def __init__(self, filter_size):
        super(feature_extractor, self).__init__()
        self.filter_size = filter_size

    def forward(self, left_input, right_input):
        """
        Feature Extractor

        Description: Aggregates the RGB values from the neighbouring pixels in the window (filter_size * filter_size).
                    No weights are learnt for this feature extractor.

        Args:
            :param left_input: Left Image
            :param right_input: Right Image

        Returns:
            :left_features: Left Image features
            :right_features: Right Image features
            :one_hot_filter: Convolution filter used to aggregate neighbour RGB features to the center pixel.
                             one_hot_filter.shape = (filter_size * filter_size)
        """
        device = left_input.get_device()
        label = torch.arange(0, self.filter_size * self.filter_size, device=device).repeat(self.filter_size * self.filter_size).view(self.filter_size * self.filter_size, 1, 1, self.filter_size, self.filter_size)
        one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
        left_features = F.conv3d(left_input.unsqueeze(1), one_hot_filter, padding=(0, self.filter_size // 2, self.filter_size // 2))
        right_features = F.conv3d(right_input.unsqueeze(1), one_hot_filter, padding=(0, self.filter_size // 2, self.filter_size // 2))
        left_features = left_features.view(left_features.size()[0], left_features.size()[1] * left_features.size()[2], left_features.size()[3], left_features.size()[4])
        right_features = right_features.view(right_features.size()[0], right_features.size()[1] * right_features.size()[2], right_features.size()[3], right_features.size()[4])
        return left_features, right_features, one_hot_filter


class Reconstruct(nn.Module):

    def __init__(self, filter_size):
        super(Reconstruct, self).__init__()
        self.filter_size = filter_size

    def forward(self, right_input, offset_x, offset_y, x_coordinate, y_coordinate, neighbour_extraction_filter):
        """
        Reconstruct the left image using the NNF(NNF represented by the offsets and the xy_coordinates)
        We did Patch Voting on the offset field, before reconstruction, in order to
                generate smooth reconstruction.
        Args:
            :right_input: Right Image
            :offset_x: horizontal offset to generate the NNF.
            :offset_y: vertical offset to generate the NNF.
            :x_coordinate: X coordinate
            :y_coordinate: Y coordinate

        Returns:
            :reconstruction: Right image reconstruction
        """
        pad_size = self.filter_size // 2
        smooth_offset_x = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))(offset_x)
        smooth_offset_y = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))(offset_y)
        smooth_offset_x = F.conv2d(smooth_offset_x, neighbour_extraction_filter, padding=(pad_size, pad_size))[:, :, pad_size:-pad_size, pad_size:-pad_size]
        smooth_offset_y = F.conv2d(smooth_offset_y, neighbour_extraction_filter, padding=(pad_size, pad_size))[:, :, pad_size:-pad_size, pad_size:-pad_size]
        coord_x = torch.clamp(x_coordinate - smooth_offset_x, min=0, max=smooth_offset_x.size()[3] - 1)
        coord_y = torch.clamp(y_coordinate - smooth_offset_y, min=0, max=smooth_offset_x.size()[2] - 1)
        coord_x -= coord_x.size()[3] / 2
        coord_x /= coord_x.size()[3] / 2
        coord_y -= coord_y.size()[2] / 2
        coord_y /= coord_y.size()[2] / 2
        grid = torch.cat((coord_x.unsqueeze(4), coord_y.unsqueeze(4)), dim=4)
        grid = grid.view(grid.size()[0] * grid.size()[1], grid.size()[2], grid.size()[3], grid.size()[4])
        reconstruction = F.grid_sample(right_input.repeat(grid.size()[0], 1, 1, 1), grid)
        reconstruction = torch.mean(reconstruction, dim=0).unsqueeze(0)
        return reconstruction


class DisparityInitialization(nn.Module):

    def __init__(self):
        super(DisparityInitialization, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_intervals=10):
        """
        PatchMatch Initialization Block
        Description:    Rather than allowing each sample/ particle to reside in the full disparity space,
                        we divide the search space into 'number_of_intervals' intervals, and force the
                        i-th particle to be in a i-th interval. This guarantees the diversity of the
                        particles and helps improve accuracy for later computations.

                        As per implementation,
                        this function divides the complete disparity search space into multiple intervals.

        Args:
            :min_disparity: Min Disparity of the disparity search range.
            :max_disparity: Max Disparity of the disparity search range.
            :number_of_intervals (default: 10): Number of samples to be generated.
        Returns:
            :interval_noise: Random value between 0-1. Represents offset of the from the interval_min_disparity.
            :interval_min_disparity: disparity_sample = interval_min_disparity + interval_noise
            :multiplier: 1.0 / number_of_intervals
        """
        device = min_disparity.get_device()
        multiplier = 1.0 / number_of_intervals
        range_multiplier = torch.arange(0.0, 1, multiplier, device=device).view(number_of_intervals, 1, 1)
        range_multiplier = range_multiplier.repeat(1, min_disparity.size()[2], min_disparity.size()[3])
        interval_noise = min_disparity.new_empty(min_disparity.size()[0], number_of_intervals, min_disparity.size()[2], min_disparity.size()[3]).uniform_(0, 1)
        interval_min_disparity = min_disparity + (max_disparity - min_disparity) * range_multiplier
        return interval_noise, interval_min_disparity, multiplier


class Evaluate(nn.Module):

    def __init__(self, filter_size=3, temperature=7):
        super(Evaluate, self).__init__()
        self.temperature = temperature
        self.filter_size = filter_size
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, left_input, right_input, disparity_samples, normalized_disparity_samples):
        """
        PatchMatch Evaluation Block
        Description:    For each pixel i, matching scores are computed by taking the inner product between the
            left feature and the right feature: score(i,j) = feature_left(i), feature_right(i+disparity(i,j))
            for all candidates j. The best k disparity value for each pixel is carried towards the next iteration.

            As per implementation,
            the complete disparity search range is discretized into intervals in
            DisparityInitialization() function. Corresponding to each disparity interval, we have multiple samples
            to evaluate. The best disparity sample per interval is the output of the function.

        Args:
            :left_input: Left Image Feature Map
            :right_input: Right Image Feature Map
            :disparity_samples: Disparity Samples to be evaluated. For each pixel, we have
                                ("number of intervals" X "number_of_samples_per_intervals") samples.

            :normalized_disparity_samples:
        Returns:
            :disparity_samples: Evaluated disparity sample, one per disparity interval.
            :normalized_disparity_samples: Evaluated normaized disparity sample, one per disparity interval.
        """
        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2]).view(left_input.size()[2], left_input.size()[3])
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)
        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        disparity_sample_strength = disparity_samples.new(disparity_samples.size()[0], disparity_samples.size()[1], disparity_samples.size()[2], disparity_samples.size()[3])
        right_y_coordinate = left_y_coordinate.expand(disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]).float()
        right_y_coordinate = right_y_coordinate - disparity_samples
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)
        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())
        disparity_sample_strength = torch.mean(left_feature_map * warped_right_feature_map, dim=1) * self.temperature
        disparity_sample_strength = disparity_sample_strength.view(disparity_sample_strength.size()[0], disparity_sample_strength.size()[1] // self.filter_size, self.filter_size, disparity_sample_strength.size()[2], disparity_sample_strength.size()[3])
        disparity_samples = disparity_samples.view(disparity_samples.size()[0], disparity_samples.size()[1] // self.filter_size, self.filter_size, disparity_samples.size()[2], disparity_samples.size()[3])
        normalized_disparity_samples = normalized_disparity_samples.view(normalized_disparity_samples.size()[0], normalized_disparity_samples.size()[1] // self.filter_size, self.filter_size, normalized_disparity_samples.size()[2], normalized_disparity_samples.size()[3])
        disparity_sample_strength = disparity_sample_strength.permute([0, 2, 1, 3, 4])
        disparity_samples = disparity_samples.permute([0, 2, 1, 3, 4])
        normalized_disparity_samples = normalized_disparity_samples.permute([0, 2, 1, 3, 4])
        disparity_sample_strength = torch.softmax(disparity_sample_strength, dim=1)
        disparity_samples = torch.sum(disparity_samples * disparity_sample_strength, dim=1)
        normalized_disparity_samples = torch.sum(normalized_disparity_samples * disparity_sample_strength, dim=1)
        return normalized_disparity_samples, disparity_samples


class Propagation(nn.Module):

    def __init__(self, filter_size=3):
        super(Propagation, self).__init__()
        self.filter_size = filter_size

    def forward(self, disparity_samples, device, propagation_type='horizontal'):
        """
        PatchMatch Propagation Block
        Description:    Particles from adjacent pixels are propagated together through convolution with a
            pre-defined one-hot filter pattern, which en-codes the fact that we allow each pixel
            to propagate particles to its 4-neighbours.

            As per implementation, the complete disparity search range is discretized into intervals in
            DisparityInitialization() function.
            Now, propagation of samples from neighbouring pixels, is done per interval. This implies that after
            propagation, number of samples per pixel = (filter_size X number_of_intervals)

        Args:
            :disparity_samples:
            :device: Cuda device
            :propagation_type (default:"horizontal"): In order to be memory efficient, we use separable convolutions
                                                    for propagtaion.

        Returns:
            :aggregated_disparity_samples: Disparity Samples aggregated from the neighbours.

        """
        disparity_samples = disparity_samples.view(disparity_samples.size()[0], 1, disparity_samples.size()[1], disparity_samples.size()[2], disparity_samples.size()[3])
        if propagation_type is 'horizontal':
            label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(self.filter_size, 1, 1, 1, self.filter_size)
            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            aggregated_disparity_samples = F.conv3d(disparity_samples, one_hot_filter, padding=(0, 0, self.filter_size // 2))
        else:
            label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(self.filter_size, 1, 1, self.filter_size, 1).long()
            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            aggregated_disparity_samples = F.conv3d(disparity_samples, one_hot_filter, padding=(0, self.filter_size // 2, 0))
        aggregated_disparity_samples = aggregated_disparity_samples.permute([0, 2, 1, 3, 4])
        aggregated_disparity_samples = aggregated_disparity_samples.contiguous().view(aggregated_disparity_samples.size()[0], aggregated_disparity_samples.size()[1] * aggregated_disparity_samples.size()[2], aggregated_disparity_samples.size()[3], aggregated_disparity_samples.size()[4])
        return aggregated_disparity_samples


class PatchMatch(nn.Module):

    def __init__(self, propagation_filter_size=3):
        super(PatchMatch, self).__init__()
        self.propagation_filter_size = propagation_filter_size
        self.propagation = Propagation(filter_size=propagation_filter_size)
        self.disparity_initialization = DisparityInitialization()
        self.evaluate = Evaluate(filter_size=propagation_filter_size)

    def forward(self, left_input, right_input, min_disparity, max_disparity, sample_count=10, iteration_count=3):
        """
        Differntail PatchMatch Block
        Description:    In this work, we unroll generalized PatchMatch as a recurrent neural network,
                        where each unrolling step is equivalent to each iteration of the algorithm.
                        This is important as it allow us to train our full model end-to-end.
                        Specifically, we design the following layers:
                            - Initialization or Paticle Sampling
                            - Propagation
                            - Evaluation
        Args:
            :left_input: Left Image feature map
            :right_input: Right image feature map
            :min_disparity: Min of the disparity search range
            :max_disparity: Max of the disparity search range
            :sample_count (default:10): Number of disparity samples per pixel. (similar to generalized PatchMatch)
            :iteration_count (default:3) : Number of PatchMatch iterations

        Returns:
            :disparity_samples: For each pixel, this function returns "sample_count" disparity samples.
        """
        device = left_input.get_device()
        min_disparity = torch.floor(min_disparity)
        max_disparity = torch.ceil(max_disparity)
        normalized_disparity_samples, min_disp_tensor, multiplier = self.disparity_initialization(min_disparity, max_disparity, sample_count)
        min_disp_tensor = min_disp_tensor.unsqueeze(2).repeat(1, 1, self.propagation_filter_size, 1, 1).view(min_disp_tensor.size()[0], min_disp_tensor.size()[1] * self.propagation_filter_size, min_disp_tensor.size()[2], min_disp_tensor.size()[3])
        for prop_iter in range(iteration_count):
            normalized_disparity_samples = self.propagation(normalized_disparity_samples, device, propagation_type='horizontal')
            disparity_samples = normalized_disparity_samples * (max_disparity - min_disparity) * multiplier + min_disp_tensor
            normalized_disparity_samples, disparity_samples = self.evaluate(left_input, right_input, disparity_samples, normalized_disparity_samples)
            normalized_disparity_samples = self.propagation(normalized_disparity_samples, device, propagation_type='vertical')
            disparity_samples = normalized_disparity_samples * (max_disparity - min_disparity) * multiplier + min_disp_tensor
            normalized_disparity_samples, disparity_samples = self.evaluate(left_input, right_input, disparity_samples, normalized_disparity_samples)
        return disparity_samples


class ImageReconstruction(nn.Module):

    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.patch_match = PatchMatch(args.patch_match_args)
        filter_size = args.feature_extractor_filter_size
        self.feature_extractor = feature_extractor(filter_size)
        self.reconstruct = Reconstruct(filter_size)

    def forward(self, left_input, right_input):
        """
        ImageReconstruction:
        Description: This class performs the task of reconstruction the left image using the data of the other image,,
            by fidning correspondences (nnf) between the two fields.
            The images acan be any random images with some overlap between the two to assist
            the correspondence matching.
            For feature_extractor, we just use the RGB features of a (self.filter_size * self.filter_size) patch
            around each pixel.
            For finding the correspondences, we use the Differentiable PatchMatch.
            ** Note: There is no assumption of rectification between the two images. **
            ** Note: The words 'left' and 'right' do not have any significance.**


        Args:
            :left_input:  Left Image (Image 1)
            :right_input:  Right Image (Image 2)

        Returns:
            :reconstruction: Reconstructed left image.
        """
        left_features, right_features, neighbour_extraction_filter = self.feature_extractor(left_input, right_input)
        offset_x, offset_y, x_coordinate, y_coordinate = self.patch_match(left_features, right_features)
        reconstruction = self.reconstruct(right_input, offset_x, offset_y, x_coordinate, y_coordinate, neighbour_extraction_filter.squeeze(1))
        return reconstruction


class RandomSampler(nn.Module):

    def __init__(self, device, number_of_samples):
        super(RandomSampler, self).__init__()
        self.number_of_samples = number_of_samples
        self.range_multiplier = torch.arange(0.0, number_of_samples + 1, 1, device=device).view(number_of_samples + 1, 1, 1)

    def forward(self, min_offset_x, max_offset_x, min_offset_y, max_offset_y):
        """
        Random Sampler:
            Given the search range per pixel (defined by: [[lx(i), ux(i)], [ly(i), uy(i)]]),
            where lx = lower_bound of the hoizontal offset,
                  ux = upper_bound of the horizontal offset,
                  ly = lower_bound of the vertical offset,
                  uy = upper_bound of teh vertical offset, for all pixel i. )
            random sampler generates samples from this search range.
            First the search range is discretized into `number_of_samples` buckets,
            then a random sample is generated from each random bucket.
            ** Discretization is done in both xy directions. ** (similar to meshgrid)

        Args:
            :min_offset_x: Min horizontal offset of the search range.
            :max_offset_x: Max horizontal offset of the search range.
            :min_offset_y: Min vertical offset of the search range.
            :max_offset_y: Max vertical offset of the search range.
        Returns:
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.
        """
        device = min_offset_x.get_device()
        noise = torch.rand(min_offset_x.repeat(1, self.number_of_samples + 1, 1, 1).size(), device=device)
        offset_x = min_offset_x + (max_offset_x - min_offset_x) / (self.number_of_samples + 1) * (self.range_multiplier + noise)
        offset_y = min_offset_y + (max_offset_y - min_offset_y) / (self.number_of_samples + 1) * (self.range_multiplier + noise)
        offset_x = offset_x.unsqueeze_(1).expand(-1, offset_y.size()[1], -1, -1, -1)
        offset_x = offset_x.contiguous().view(offset_x.size()[0], offset_x.size()[1] * offset_x.size()[2], offset_x.size()[3], offset_x.size()[4])
        offset_y = offset_y.unsqueeze_(2).expand(-1, -1, offset_y.size()[1], -1, -1)
        offset_y = offset_y.contiguous().view(offset_y.size()[0], offset_y.size()[1] * offset_y.size()[2], offset_y.size()[3], offset_y.size()[4])
        return offset_x, offset_y


class PropagationFaster(nn.Module):

    def __init__(self):
        super(PropagationFaster, self).__init__()

    def forward(self, offset_x, offset_y, device, propagation_type='horizontal'):
        """
        Faster version of PatchMatch Propagation Block
        This version uses a fixed propagation filter size of size 3. This implementation is not recommended
        and is used only to do the propagation faster.

        Description:    Particles from adjacent pixels are propagated together through convolution with a
                        one-hot filter, which en-codes the fact that we allow each pixel
                        to propagate particles to its 4-neighbours.
        Args:
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.
            :device: Cuda/ CPU device
            :propagation_type (default:"horizontal"): In order to be memory efficient, we use separable convolutions
                                                    for propagtaion.

        Returns:
            :aggregated_offset_x: Horizontal offset samples aggregated from the neighbours.
            :aggregated_offset_y: Vertical offset samples aggregated from the neighbours.

        """
        self.vertical_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], 1, offset_x.size()[3]))
        self.horizontal_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], offset_x.size()[2], 1))
        if propagation_type is 'horizontal':
            offset_x = torch.cat((torch.cat((self.horizontal_zeros, offset_x[:, :, :, :-1]), dim=3), offset_x, torch.cat((offset_x[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)
            offset_y = torch.cat((torch.cat((self.horizontal_zeros, offset_y[:, :, :, :-1]), dim=3), offset_y, torch.cat((offset_y[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)
        else:
            offset_x = torch.cat((torch.cat((self.vertical_zeros, offset_x[:, :, :-1, :]), dim=2), offset_x, torch.cat((offset_x[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)
            offset_y = torch.cat((torch.cat((self.vertical_zeros, offset_y[:, :, :-1, :]), dim=2), offset_y, torch.cat((offset_y[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)
        return offset_x, offset_y


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


def convbn_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = convbn_relu(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn_relu(3, 32, 3, 2, 1, 1), convbn_relu(32, 32, 3, 1, 1, 1), convbn_relu(32, 32, 3, 1, 1, 1))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)), convbn_relu(128, 32, 1, 1, 0, 1))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)), convbn_relu(128, 32, 1, 1, 0, 1))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)), convbn_relu(128, 32, 1, 1, 0, 1))
        self.lastconv = nn.Sequential(convbn_relu(352, 128, 3, 1, 1, 1), nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, input):
        """
        Feature Extractor
        Description:    The goal of the feature extraction network is to produce a reliable pixel-wise
                        feature representation from the input image. Specifically, we employ four residual blocks
                        and use X2 dilated convolution for the last block to enlarge the receptive field.
                        We then apply spatial pyramid pooling to build a 4-level pyramid feature.
                        Through multi-scale information, the model is able to capture large context while
                        maintaining a high spatial resolution. The size of the final feature map is 1/4 of
                        the originalinput image size. We share the parameters for the left and right feature network.

        Args:
            :input: Input image (RGB)

        Returns:
            :output_feature: spp_features (downsampled X8)
            :output_raw: features (downsampled X4)
            :output1: low_level_features (downsampled X2)
        """
        output0 = self.firstconv(input)
        output1 = self.layer1(output0)
        output_raw = self.layer2(output1)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_feature = torch.cat((output, output_skip, output_branch4, output_branch3, output_branch2), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature, output_raw, output1


class SubModule(nn.Module):

    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def convbn_2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))


class RefinementNet(SubModule):

    def __init__(self, inplanes):
        super(RefinementNet, self).__init__()
        self.conv1 = nn.Sequential(convbn_2d_lrelu(inplanes, 32, kernel_size=3, stride=1, pad=1), convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1), convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1), convbn_2d_lrelu(32, 16, kernel_size=3, stride=1, pad=2, dilation=2), convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=4, dilation=4), convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=1, dilation=1))
        self.classif1 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.weight_init()

    def forward(self, input, disparity):
        """
        Refinement Block
        Description:    The network takes left image convolutional features from the second residual block
                        of the feature network and the current disparity estimation as input.
                        It then outputs the finetuned disparity prediction. The low-level feature
                        information serves as a guidance to reduce noise and improve the quality of the final
                        disparity map, especially on sharp boundaries.

        Args:
            :input: Input features composed of left image low-level features, cost-aggregator features, and
                    cost-aggregator disparity.

            :disparity: predicted disparity
        """
        output0 = self.conv1(input)
        output0 = self.classif1(output0)
        output = self.relu(output0 + disparity)
        return output


def convbn_3d_lrelu(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=(pad, pad, pad), stride=(1, stride, stride), bias=False), nn.BatchNorm3d(out_planes), nn.LeakyReLU(0.1, inplace=True))


def convbn_transpose_3d(inplanes, outplanes, kernel_size, padding, output_padding, stride, bias):
    return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size, padding=padding, output_padding=output_padding, stride=stride, bias=bias), nn.BatchNorm3d(outplanes))


class HourGlass(SubModule):

    def __init__(self, inplanes=16):
        super(HourGlass, self).__init__()
        self.conv1 = convbn_3d_lrelu(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = convbn_3d_lrelu(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.conv1_1 = convbn_3d_lrelu(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = convbn_3d_lrelu(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1)
        self.conv3 = convbn_3d_lrelu(inplanes * 4, inplanes * 8, kernel_size=3, stride=2, pad=1)
        self.conv4 = convbn_3d_lrelu(inplanes * 8, inplanes * 8, kernel_size=3, stride=1, pad=1)
        self.conv5 = convbn_transpose_3d(inplanes * 8, inplanes * 4, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv6 = convbn_transpose_3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv7 = convbn_transpose_3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.last_conv3d_layer = nn.Sequential(convbn_3d_lrelu(inplanes, inplanes * 2, 3, 1, 1), nn.Conv3d(inplanes * 2, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.softmax = nn.Softmax(dim=1)
        self.weight_init()


class MaxDisparityPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(MaxDisparityPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input, input_disparity):
        """
        Confidence Range Prediction (Max Disparity):
        Description:    The network has a convolutional encoder-decoder structure. It takes the sparse
                disparity estimations from the differentiable PatchMatch, the left image and the warped right image
                (warped according to the sparse disparity estimations) as input and outputs the upper bound of
                the confidence range for each pixel i.
        Args:
            :input: Left and Warped right Image features as Cost Volume.
            :input_disparity: PatchMatch predicted disparity samples.
        Returns:
            :disparity_output: Max Disparity of the reduced disaprity search range.
            :feature_output:   High-level features of the MaxDisparityPredictor
        """
        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0
        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0
        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0
        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1)
        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2
        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1).unsqueeze(1)
        return disparity_output, feature_output


class MinDisparityPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(MinDisparityPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input, input_disparity):
        """
        Confidence Range Prediction (Min Disparity):
        Description:    The network has a convolutional encoder-decoder structure. It takes the sparse
                disparity estimations from the differentiable PatchMatch, the left image and the warped right image
                (warped according to the sparse disparity estimations) as input and outputs the lower bound of
                the confidence range for each pixel i.
        Args:
            :input: Left and Warped right Image features as Cost Volume.
            :input_disparity: PatchMatch predicted disparity samples.
        Returns:
            :disparity_output: Min Disparity of the reduced disaprity search range.
            :feature_output:   High-level features of the MaxDisparityPredictor
        """
        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0
        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0
        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0
        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1)
        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2
        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1).unsqueeze(1)
        return disparity_output, feature_output


class CostAggregator(HourGlass):

    def __init__(self, cost_aggregator_inplanes, hourglass_inplanes=16):
        super(CostAggregator, self).__init__(inplanes=16)
        self.dres0 = nn.Sequential(convbn_3d_lrelu(cost_aggregator_inplanes, 64, 3, 1, 1), convbn_3d_lrelu(64, 32, 3, 1, 1))
        self.dres1 = nn.Sequential(convbn_3d_lrelu(32, 32, 3, 1, 1), convbn_3d_lrelu(32, hourglass_inplanes, 3, 1, 1))

    def forward(self, input, input_disparity):
        """
        3D Cost Aggregator
        Description:    Based on the predicted range in the pruning module,
                we build the 3D cost volume estimator and conduct spatial aggregation.
                Following common practice, we take the left image, the warped right image and corresponding disparities
                as input and output the cost over the disparity range at the size B X R X H X W , where R is the number
                of disparities per pixel. Compared to prior work, our R is more than 10 times smaller, making
                this module very efficient. Soft-arg max is again used to predict the disparity value ,
                so that our approach is end-to-end trainable.

        Args:
            :input:   Cost-Volume composed of left image features, warped right image features,
                      Confidence range Predictor features and input disparity samples/

            :input_disparity: input disparity samples.

        Returns:
            :disparity_output: Predicted disparity
            :feature_output: High-level features of 3d-Cost Aggregator

        """
        output0 = self.dres0(input)
        output0_b = self.dres1(output0)
        output0 = self.conv1(output0_b)
        output0_a = self.conv2(output0) + output0
        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0
        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0
        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1) + output0_b
        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2
        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1)
        return disparity_output.unsqueeze(1), feature_output


class UniformSampler(nn.Module):

    def __init__(self):
        super(UniformSampler, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_samples=10):
        """
        Uniform Sampler
        Description:    The Confidence Range Predictor predicts a reduced disparity search range R(i) = [l(i), u(i)]
            for each pixel i. We then, generate disparity samples from this reduced search range for Cost Aggregation
            or second stage of Patch Match. From experiments, we found Uniform sampling to work better.

        Args:
            :min_disparity: lower bound of disparity search range (predicted by Confidence Range Predictor)
            :max_disparity: upper bound of disparity range predictor (predicted by Confidence Range Predictor)
            :number_of_samples (default:10): number of samples to be genearted.
        Returns:
            :sampled_disparities: Uniformly generated disparity samples from the input search range.
        """
        device = min_disparity.get_device()
        multiplier = (max_disparity - min_disparity) / (number_of_samples + 1)
        range_multiplier = torch.arange(1.0, number_of_samples + 1, 1, device=device).view(number_of_samples, 1, 1)
        sampled_disparities = min_disparity + multiplier * range_multiplier
        return sampled_disparities


class SpatialTransformer(nn.Module):

    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and teh disparity samples, generates:
                    - Per sample cost as <left_image_features, right_image_features>, <.,.> denotes scalar-product.
                    - Warped righ image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples genearted by PatchMatch

        Returns:
            :disparity_samples_strength_1: Cost associated with each disaprity sample.
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """
        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)
        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        disparity_samples = disparity_samples.float()
        right_y_coordinate = left_y_coordinate.expand(disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples
        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)
        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())
        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) + (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * warped_right_feature_map + torch.zeros_like(warped_right_feature_map)
        return warped_right_feature_map, left_feature_map


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PropagationFaster,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RefinementNet,
     lambda: ([], {'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
]

class Test_uber_research_DeepPruner(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

