import sys
_module = sys.modules[__name__]
del sys
setup = _module
vtunet = _module
configuration = _module
evaluation = _module
add_dummy_task_with_mean_over_all_tasks = _module
add_mean_dice_to_json = _module
collect_results_files = _module
evaluator = _module
metrics = _module
model_selection = _module
collect_all_fold0_results_and_summarize_in_one_csv = _module
ensemble = _module
figure_out_what_to_submit = _module
rank_candidates = _module
rank_candidates_StructSeg = _module
rank_candidates_cascade = _module
summarize_results_in_one_json = _module
summarize_results_with_plans = _module
region_based_evaluation = _module
surface_dice = _module
DatasetAnalyzer = _module
experiment_planning = _module
change_batch_size = _module
common_utils = _module
experiment_planner_baseline_2DUNet = _module
experiment_planner_baseline_2DUNet_v21 = _module
experiment_planner_baseline_3DUNet = _module
experiment_planner_baseline_3DUNet_v21 = _module
summarize_plans = _module
utils = _module
vtunet_convert_decathlon_task = _module
vtunet_plan_and_preprocess = _module
inference = _module
change_trainer = _module
ensemble_predictions = _module
predict = _module
predict_simple = _module
pretrained_models = _module
collect_pretrained_models = _module
download_pretrained_model = _module
predict = _module
predict_simple = _module
segmentation_export = _module
inference_tumor = _module
network_architecture = _module
generic_UNet = _module
initialization = _module
neural_network = _module
vtunet_tumor = _module
paths = _module
connected_components = _module
consolidate_all_for_paper = _module
consolidate_postprocessing = _module
consolidate_postprocessing_simple = _module
cropping = _module
preprocessor_scale_RGB_to_0_1 = _module
preprocessing = _module
sanity_checks = _module
init = _module
run = _module
default_configuration = _module
load_pretrained_weights = _module
run_training = _module
training = _module
cascade_stuff = _module
predict_next_stage = _module
data_augmentation = _module
custom_transforms = _module
data_augmentation_moreDA = _module
data_augmentation_noDA = _module
default_data_augmentation = _module
downsampling = _module
pyramid_augmentations = _module
dataloading = _module
dataset_loading = _module
poly_lr = _module
TopK_loss = _module
loss_functions = _module
crossentropy = _module
deep_supervision = _module
dice_loss = _module
model_restore = _module
network_training = _module
network_trainer = _module
vtunetTrainer = _module
vtunetTrainerV2 = _module
vtunetTrainerV2_vtunet_tumor = _module
vtunetTrainerV2_vtunet_tumor_base = _module
ranger = _module
utilities = _module
distributed = _module
file_conversions = _module
file_endings = _module
folder_names = _module
nd_softmax = _module
one_hot_encoding = _module
overlay_plots = _module
random_stuff = _module
recursive_delete_npz = _module
recursive_rename_taskXX_to_taskXXX = _module
sitk_stuff = _module
task_name_id_conversion = _module
tensor_utilities = _module
to_torch = _module
config = _module
dataset = _module
batch_utils = _module
brats = _module
image_utils = _module
loss = _module
dice = _module
test = _module
train = _module
utils = _module
vision_transformer = _module
vt_unet = _module

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


from copy import deepcopy


from typing import Tuple


from typing import Union


from typing import List


import numpy as np


import torch


from torch import nn


import torch.nn.functional


from scipy.ndimage.filters import gaussian_filter


from torch.cuda.amp import autocast


import copy


import logging


from functools import reduce


from functools import lru_cache


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.checkpoint as checkpoint


from torch.nn.functional import avg_pool2d


from torch.nn.functional import avg_pool3d


from torch import Tensor


import matplotlib


from sklearn.model_selection import KFold


from torch.cuda.amp import GradScaler


from torch.optim.lr_scheduler import _LRScheduler


from time import time


from time import sleep


from torch.optim import lr_scheduler


import matplotlib.pyplot as plt


from collections import OrderedDict


import torch.backends.cudnn as cudnn


from abc import abstractmethod


import math


from torch.optim.optimizer import Optimizer


from torch import distributed


from torch import autograd


from torch.nn.parallel import DistributedDataParallel as DDP


import random


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.dataset import Dataset


import pandas as pd


import torch.nn.parallel


import torch.optim


import torch.utils.data


from torch.utils.tensorboard import SummaryWriter


import time


from torch.autograd import Variable


from matplotlib import pyplot as plt


from numpy import logical_and as l_and


from numpy import logical_not as l_not


from torch import distributed as dist


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels, conv_op=nn.Conv2d, conv_kwargs=None, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 0.01, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-05, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):

    def __init__(self, input_feature_channels, output_feature_channels, num_convs, conv_op=nn.Conv2d, conv_kwargs=None, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        """
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        """
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 0.01, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-05, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(*([basic_block(input_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs_first_conv, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs)] + [basic_block(output_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == 'cpu':
            return 'cpu'
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == 'cpu':
            self.cpu()
        else:
            self

    def forward(self, x):
        raise NotImplementedError


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [(maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i) for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


class no_op(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i for i in data]
    else:
        data = data
    return data


class SegmentationNetwork(NeuralNetwork):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        self.inference_apply_nonlin = lambda x: x
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...]=(0, 1, 2), use_sliding_window: bool=False, step_size: float=0.5, patch_size: Tuple[int, ...]=None, regions_class_order: Tuple[int, ...]=None, use_gaussian: bool=False, pad_border_mode: str='constant', pad_kwargs: dict=None, all_in_gpu: bool=False, verbose: bool=True, mixed_precision: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the himashi to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()
        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive predictions'
        if verbose:
            None
        assert self.get_device() != 'cpu', 'CPU not implemented'
        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError('mirror axes. duh')
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError('mirror axes. duh')
        if self.training:
            None
        assert len(x.shape) == 4, 'data must have shape (c,x,y,z)'
        if mixed_precision:
            context = autocast
        else:
            context = no_op
        with context():
            with torch.no_grad():
                res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose)
        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1.0 / 8) ->np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [(i // 2) for i in patch_size]
        sigmas = [(i * sigma_scale) for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])
        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) ->List[List[int]]:
        assert [(i >= j) for i, j in zip(image_size, patch_size)], 'image size must be as large or larger than patch_size'
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
        target_step_sizes_in_voxels = [(i * step_size) for i in patch_size]
        num_steps = [(int(np.ceil((i - k) / j)) + 1) for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]
        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
            steps.append(steps_here)
        return steps

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple, patch_size: tuple, regions_class_order: tuple, use_gaussian: bool, pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool, verbose: bool) ->Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, 'x must be (c, x, y, z)'
        assert self.get_device() != 'cpu'
        if verbose:
            None
        if verbose:
            None
        assert patch_size is not None, 'patch_size cannot be None for tiled prediction'
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
        if verbose:
            None
            None
            None
            None
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all([(i == j) for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose:
                    None
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1.0 / 8)
                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose:
                    None
                gaussian_importance_map = self._gaussian_3d
            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
        else:
            gaussian_importance_map = None
        if all_in_gpu:
            if use_gaussian and num_tiles > 1:
                gaussian_importance_map = gaussian_importance_map.half()
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()
                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())
            if verbose:
                None
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
            if verbose:
                None
            data = torch.from_numpy(data)
            if verbose:
                None
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring, gaussian_importance_map)[0]
                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
        slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        class_probabilities = aggregated_results / aggregated_nb_of_predictions
        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c
        if all_in_gpu:
            if verbose:
                None
            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            class_probabilities = class_probabilities.detach().cpu().numpy()
        if verbose:
            None
        return predicted_segmentation, class_probabilities

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple=(0, 1, 2), regions_class_order: tuple=None, pad_border_mode: str='constant', pad_kwargs: dict=None, verbose: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, 'x must be (c, x, y)'
        assert self.get_device() != 'cpu'
        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to run _internal_predict_2D_2Dconv'
        if verbose:
            None
        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)
        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring, None)[0]
        slicer = tuple([slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) - (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]
        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c
        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_3Dconv(self, x: np.ndarray, min_size: Tuple[int, ...], do_mirroring: bool, mirror_axes: tuple=(0, 1, 2), regions_class_order: tuple=None, pad_border_mode: str='constant', pad_kwargs: dict=None, verbose: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, 'x must be (c, x, y, z)'
        assert self.get_device() != 'cpu'
        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to run _internal_predict_3D_3Dconv'
        if verbose:
            None
        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)
        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(data[None], mirror_axes, do_mirroring, None)[0]
        slicer = tuple([slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) - (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]
        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c
        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple, do_mirroring: bool=True, mult: (np.ndarray or torch.tensor)=None) ->torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)
        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())
        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1
        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred
            if m == 1 and 2 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4,))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))
            if m == 2 and 1 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))
            if m == 3 and 2 in mirror_axes and 1 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))
            if m == 4 and 0 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))
            if m == 5 and 0 in mirror_axes and 2 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))
            if m == 6 and 0 in mirror_axes and 1 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))
            if m == 7 and 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))
        if mult is not None:
            result_torch[:, :] *= mult
        return result_torch

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple, do_mirroring: bool=True, mult: (np.ndarray or torch.tensor)=None) ->torch.tensor:
        assert len(x.shape) == 4, 'x must be (b, c, x, y)'
        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)
        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())
        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1
        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred
            if m == 1 and 1 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))
            if m == 2 and 0 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))
            if m == 3 and 0 in mirror_axes and 1 in mirror_axes:
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))
        if mult is not None:
            result_torch[:, :] *= mult
        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple, patch_size: tuple, regions_class_order: tuple, use_gaussian: bool, pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool, verbose: bool) ->Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 3, 'x must be (c, x, y)'
        assert self.get_device() != 'cpu'
        if verbose:
            None
        if verbose:
            None
        assert patch_size is not None, 'patch_size cannot be None for tiled prediction'
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])
        if verbose:
            None
            None
            None
            None
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all([(i == j) for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose:
                    None
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1.0 / 8)
                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose:
                    None
                gaussian_importance_map = self._gaussian_2d
            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
        else:
            gaussian_importance_map = None
        if all_in_gpu:
            if use_gaussian and num_tiles > 1:
                gaussian_importance_map = gaussian_importance_map.half()
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()
                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())
            if verbose:
                None
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
            if verbose:
                None
            data = torch.from_numpy(data)
            if verbose:
                None
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                predicted_patch = self._internal_maybe_mirror_and_pred_2D(data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring, gaussian_importance_map)[0]
                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()
                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds
        slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        class_probabilities = aggregated_results / aggregated_nb_of_predictions
        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c
        if all_in_gpu:
            if verbose:
                None
            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            class_probabilities = class_probabilities.detach().cpu().numpy()
        if verbose:
            None
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple=(0, 1), regions_class_order: tuple=None, pad_border_mode: str='constant', pad_kwargs: dict=None, all_in_gpu: bool=False, verbose: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, 'data must be c, x, y, z'
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def predict_3D_pseudo3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple=(0, 1), regions_class_order: tuple=None, pseudo3D_slices: int=5, all_in_gpu: bool=False, pad_border_mode: str='constant', pad_kwargs: dict=None, verbose: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, 'data must be c, x, y, z'
        assert pseudo3D_slices % 2 == 1, 'pseudo3D_slices must be odd'
        extra_slices = (pseudo3D_slices - 1) // 2
        shp_for_pad = np.array(x.shape)
        shp_for_pad[1] = extra_slices
        pad = np.zeros(shp_for_pad, dtype=np.float32)
        data = np.concatenate((pad, x, pad), 1)
        predicted_segmentation = []
        softmax_pred = []
        for s in range(extra_slices, data.shape[1] - extra_slices):
            d = data[:, s - extra_slices:s + extra_slices + 1]
            d = d.reshape((-1, d.shape[-2], d.shape[-1]))
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(d, min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv_tiled(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple=(0, 1), step_size: float=0.5, regions_class_order: tuple=None, use_gaussian: bool=False, pad_border_mode: str='edge', pad_kwargs: dict=None, all_in_gpu: bool=False, verbose: bool=True) ->Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, 'data must be c, x, y, z'
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, prev_v=None, prev_k=None, prev_q=None, is_decoder=False):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x2 = None
        if is_decoder:
            q = q * self.scale
            attn2 = q @ prev_k.transpose(-2, -1)
            attn2 = attn2 + relative_position_bias.unsqueeze(0)
            if mask is not None:
                nW = mask.shape[0]
                attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn2 = attn2.view(-1, self.num_heads, N, N)
                attn2 = self.softmax(attn2)
            else:
                attn2 = self.softmax(attn2)
            attn2 = self.attn_drop(attn2)
            x2 = (attn2 @ prev_v).transpose(1, 2).reshape(B_, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)
        return x, x2, v, k, q


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError('The input tensor has to be 5d!')
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        sin_inp_z = torch.einsum('i,j->ij', pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z
        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[2], 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows, cross_attn_windows, v, k, q = self.attn(x_windows, mask=attn_mask, prev_v=prev_v, prev_k=prev_k, prev_q=prev_q, is_decoder=is_decoder)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x2 = None
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        if cross_attn_windows is not None:
            cross_attn_windows = cross_attn_windows.view(-1, *(window_size + (C,)))
            cross_shifted_x = window_reverse(cross_attn_windows, window_size, B, Dp, Hp, Wp)
            if any(i > 0 for i in shift_size):
                x2 = torch.roll(cross_shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                x2 = cross_shifted_x
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x2 = x2[:, :D, :H, :W, :].contiguous()
        return x, x2, v, k, q

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward_part3(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder=False):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        alpha = 0.5
        shortcut = x
        x2, v, k, q = None, None, None, None
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x, x2, v, k, q = self.forward_part1(x, mask_matrix, prev_v, prev_k, prev_q, is_decoder)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        if x2 is not None:
            x2 = shortcut + self.drop_path(x2)
            if self.use_checkpoint:
                x2 = x2 + checkpoint.checkpoint(self.forward_part2, x2)
            else:
                x2 = x2 + self.forward_part2(x2)
            FPE = PositionalEncoding3D(x.shape[4])
            x = torch.add((1 - alpha) * x, alpha * x2) + self.forward_part3(FPE(x))
        return x, v, k, q


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand_Up(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, 32, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // 4)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PatchExpand(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, D * 8, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // 4)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class FinalPatchExpand_X4(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.permute(0, 4, 1, 2, 3)
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale, c=C // self.dim_scale ** 3)
        x = self.norm(x)
        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size tuple(int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(7, 7, 7), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint) for i in range(depth)])
        if upsample is not None:
            self.upsample = PatchExpand_Up(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, prev_v1, prev_k1, prev_q1, prev_v2, prev_k2, prev_q2):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x, _, _, _ = blk(x, attn_mask, prev_v1, prev_k1, prev_q1, True)
            else:
                x, _, _, _ = blk(x, attn_mask, prev_v2, prev_k2, prev_q2, True)
        if self.upsample is not None:
            x = x.permute(0, 4, 1, 2, 3)
            x = self.upsample(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, depths, num_heads, window_size=(1, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint) for i in range(depth)])
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, block_num):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        v1, k1, q1, v2, k2, q2 = None, None, None, None, None, None
        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x, v1, k1, q1 = blk(x, attn_mask, None, None, None)
            else:
                x, v2, k2, q2 = blk(x, attn_mask, None, None, None)
        x = x.reshape(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x, v1, k1, q1, v2, k2, q2


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class SwinTransformerSys3D(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple(int)): Window size. Default: (7,7,7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrained=None, pretrained2d=True, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=4, num_classes=3, embed_dim=96, depths=[2, 2, 2, 1], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=(7, 7, 7), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, frozen_stages=-1, final_upsample='expand_first', **kwargs):
        super().__init__()
        None
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], depths=depths, num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], drop_path_rate=drop_path_rate, norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), bias=False) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[1] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[2] // 2 ** (self.num_layers - 1 - i_layer)), dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), input_resolution=(patches_resolution[0] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[1] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[2] // 2 ** (self.num_layers - 1 - i_layer)), depth=depths[self.num_layers - 1 - i_layer], num_heads=num_heads[self.num_layers - 1 - i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - 1 - i_layer + 1])], norm_layer=norm_layer, upsample=PatchExpand if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == 'expand_first':
            None
            self.up = FinalPatchExpand_X4(input_resolution=(img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]), dim_scale=4, dim=embed_dim)
            self.output = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self._freeze_stages()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        v_values_1 = []
        k_values_1 = []
        q_values_1 = []
        v_values_2 = []
        k_values_2 = []
        q_values_2 = []
        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            x, v1, k1, q1, v2, k2, q2 = layer(x, i)
            v_values_1.append(v1)
            k_values_1.append(k1)
            q_values_1.append(q1)
            v_values_2.append(v2)
            k_values_2.append(k2)
            q_values_2.append(q2)
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        return x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2

    def forward_up_features(self, x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], 1)
                B, C, D, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                _, _, C = x.shape
                x = x.view(B, D, H, W, C)
                x = x.permute(0, 4, 1, 2, 3)
                x = layer_up(x, v_values_1[3 - inx], k_values_1[3 - inx], q_values_1[3 - inx], v_values_2[3 - inx], k_values_2[3 - inx], q_values_2[3 - inx])
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        D, H, W = self.patches_resolution
        B, _, _, _, C = x.shape
        if self.final_upsample == 'expand_first':
            x = self.up(x)
            x = x.view(B, 4 * D, 4 * H, 4 * W, -1)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.output(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        relative_position_index_keys = [k for k in state_dict.keys() if 'relative_position_index' in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in attn_mask_keys:
            del state_dict[k]
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1, self.patch_size[0], 1, 1) / self.patch_size[0]
        relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                None
            elif L1 != L2:
                S1 = int(L1 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1), mode='bicubic')
                relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)
        msg = self.load_state_dict(state_dict, strict=False)
        None
        None
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            None
            if self.pretrained2d:
                self.inflate_weights()
            else:
                load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2 = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2)
        x = self.up_x4(x)
        return x


class VTUNet(nn.Module):

    def __init__(self, config, num_classes=3, zero_head=False, embed_dim=96, win_size=7):
        super(VTUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = self.win_size, self.win_size, self.win_size
        self.swin_unet = SwinTransformerSys3D(img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=4, num_classes=self.num_classes, embed_dim=self.embed_dim, depths=[2, 2, 2, 1], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=self.win_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, frozen_stages=-1, final_upsample='expand_first')

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            None
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if 'model' not in pretrained_dict:
                None
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if 'output' in k:
                        None
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            None
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if 'layers.' in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = 'layers_up.' + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        None
                        del full_dict[k]
            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            None


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class MultipleOutputLoss2(nn.Module):

    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), 'x must be either tuple or list'
        assert isinstance(y, (tuple, list)), 'y must be either tuple or list'
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))
        if all([(i == j) for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == 'cuda':
                y_onehot = y_onehot
            y_onehot.scatter_(1, gt, 1)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)
    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2
    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


class GDL(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0, square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()
        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        if all([(i == j) for i, j in zip(x.shape, y.shape)]):
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == 'cuda':
                y_onehot = y_onehot
            y_onehot.scatter_(1, gt, 1)
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)
        volumes = sum_tensor(y_onehot, axes) + 1e-06
        if self.square_volumes:
            volumes = volumes ** 2
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1
        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return -dc


class SoftDiceLoss(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0):
        """
        """
        super(SoftDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth
        dc = nominator / (denominator + 1e-08)
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc


class MCCLoss(nn.Module):

    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])
        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels
        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth
        mcc = nominator / denominator
        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()
        return -mcc


class SoftDiceLossSquared(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if all([(i == j) for i, j in zip(x.shape, y.shape)]):
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == 'cuda':
                    y_onehot = y_onehot
                y_onehot.scatter_(1, y, 1).float()
        intersect = x * y_onehot
        denominator = x ** 2 + y_onehot ** 2
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth
        dc = 2 * intersect / denominator
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc


softmax_helper = lambda x: F.softmax(x, 1)


class DC_and_CE_loss(nn.Module):

    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate='sum', square_dice=False, weight_ce=1.0, weight_dice=1.0, log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ignore_label = ignore_label
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None
        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        if self.aggregate == 'sum':
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class DC_and_BCE_loss(nn.Module):

    def __init__(self, bce_kwargs, soft_dice_kwargs, aggregate='sum'):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)
        if self.aggregate == 'sum':
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class GDL_and_CE_loss(nn.Module):

    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate='sum'):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == 'sum':
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class TopKLoss(RobustCrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1,)), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class DC_and_topk_loss(nn.Module):

    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate='sum', square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == 'sum':
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ['ET', 'TC', 'WT']
        self.device = 'cpu'

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.0
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)
        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                None
                if inputs.sum() == 0:
                    return torch.tensor(1.0, device='cuda')
                else:
                    return torch.tensor(0.0, device='cuda')
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = 2 * intersection / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        CE_L = torch.nn.BCELoss()
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        final_dice = (0.7 * dice + 0.3 * ce) / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class EDiceLoss_Val(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss_Val, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ['ET', 'TC', 'WT']
        self.device = 'cpu'

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.0
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)
        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                None
                if inputs.sum() == 0:
                    return torch.tensor(1.0, device='cuda')
                else:
                    return torch.tensor(0.0, device='cuda')
        intersection = EDiceLoss_Val.compute_intersection(inputs, targets)
        if metric_mode:
            dice = 2 * intersection / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class Merge_Block(nn.Module):

    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        D = 32
        H = W = int(np.sqrt(new_HW // D))
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvDropoutNonlinNorm,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvDropoutNormNonlin,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EDiceLoss_Val,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GDL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MCCLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultipleOutputLoss2,
     lambda: ([], {'loss': MSELoss()}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), (torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     True),
    (PatchEmbed3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SoftDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftDiceLossSquared,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (StackedConvLayers,
     lambda: ([], {'input_feature_channels': 4, 'output_feature_channels': 4, 'num_convs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TopKLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_himashi92_VT_UNet(_paritybench_base):
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

