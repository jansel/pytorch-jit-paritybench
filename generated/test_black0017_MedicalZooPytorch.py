import sys
_module = sys.modules[__name__]
del sys
test_miccai_2019 = _module
train_brats2018_new = _module
train_brats2019_new = _module
train_brats2020_new = _module
train_iseg2017_new = _module
train_iseg2019_new = _module
train_mrbrains_4_classes = _module
train_mrbrains_9_classes = _module
lib = _module
augment3D = _module
elastic_deform = _module
gaussian_noise = _module
random_crop = _module
random_flip = _module
random_rescale = _module
random_rotate = _module
random_shift = _module
BCE_dice = _module
BaseClass = _module
ContrastiveLoss = _module
Dice2D = _module
VAEloss = _module
losses3D = _module
basic = _module
dice = _module
generalized_dice = _module
pixel_wise_cross_entropy = _module
tags_angular_loss = _module
weight_cross_entropy = _module
weight_smooth_l1 = _module
COVIDxdataset = _module
Covid_Segmentation_dataset = _module
medloaders = _module
brats2018 = _module
brats2019 = _module
brats2020 = _module
covid_ct_dataset = _module
iseg2017 = _module
iseg2019 = _module
ixi_t1_t2 = _module
medical_image_process = _module
medical_loader_utils = _module
miccai_2019_pathology = _module
mrbrains2018 = _module
BaseModelClass = _module
COVIDNet = _module
DenseVoxelNet = _module
Densenet3D = _module
HighResNet3D = _module
HyperDensenet = _module
ResNet3DMedNet = _module
ResNet3D_VAE = _module
SkipDenseNet3D = _module
Unet2D = _module
Unet3D = _module
Vnet = _module
medzoo = _module
BaseTrainer = _module
train = _module
train_covid = _module
train_old = _module
trainer = _module
utils = _module
covid_utils = _module
general = _module
save_old = _module
writer_old = _module
BaseWriter = _module
visual3D_temp = _module
conf_matrix = _module
viz = _module
viz_2d = _module
viz_old = _module
writer = _module
inference = _module
test_augmentations = _module
test_basewriter = _module
test_covid_ct = _module
test_dataloaders = _module
test_losses3D = _module
test_medical_3D_augemt = _module
test_medical_3D_preprocessing = _module
test_models = _module
test_vizual = _module
train_with_trainer_class = _module

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


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


from torch import nn as nn


import torch.functional as F


from torch.nn import MSELoss


from torch.nn import SmoothL1Loss


from torch.nn import L1Loss


from torch.utils.data import Dataset


from torchvision import transforms


import numpy as np


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


from scipy import ndimage


from abc import ABC


from abc import abstractmethod


import torch.nn.functional as F


from torchvision import models


from functools import partial


from collections import OrderedDict


import torch.optim as optim


from numpy import inf


import random


import time


import torch.backends.cudnn as cudnn


import itertools


import matplotlib.pyplot as plt


import math


def expand_as_one_hot(target, classes):
    shape = target.size()
    shape = list(shape)
    shape.insert(1, classes)
    shape = tuple(shape)
    src = target.unsqueeze(1).long()
    return torch.zeros(shape).scatter_(1, src, 1).squeeze(0)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.classes = None
        self.skip_index_after = None
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = expand_as_one_hot(target.long(), self.classes)
        assert input.dim() == target.dim() == 5, "'input' and 'target' have different number of dims"
        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            None
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        input = self.normalization(input)
        per_channel_dice = self.dice(input, target, weight=self.weight)
        loss = 1.0 - torch.mean(per_channel_dice)
        per_channel_dice = per_channel_dice.detach().cpu().numpy()
        return loss, per_channel_dice


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-06, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=4, skip_index_after=None, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss(classes=classes)
        self.classes = classes

    def forward(self, input, target):
        target_expanded = expand_as_one_hot(target.long(), self.classes)
        assert input.size() == target_expanded.size(), "'input' and 'target' must have the same shape"
        loss_1 = self.alpha * self.bce(input, target_expanded)
        loss_2, channel_score = self.beta * self.dice(input, target_expanded)
        return loss_1 + loss_2, channel_score


class ContrastiveLoss(torch.nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm='fro', alpha=1.0, beta=1.0, gamma=0.001):
        super(ContrastiveLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_cluster_means(self, input, target):
        embedding_dims = input.size()[1]
        target = target.unsqueeze(2)
        target_copy = target.clone()
        shape = list(target.size())
        shape[2] = embedding_dims
        target = target.expand(shape)
        input = input.unsqueeze(1)
        embeddings_per_instance = input * target
        num = torch.sum(embeddings_per_instance, dim=(3, 4, 5), keepdim=True)
        num_voxels_per_instance = torch.sum(target_copy, dim=(3, 4, 5), keepdim=True)
        mean_embeddings = num / num_voxels_per_instance
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target):
        embedding_norms = torch.norm(embeddings_per_instance - cluster_means, self.norm, dim=2)
        embedding_norms = embedding_norms * target
        embedding_variance = torch.clamp(embedding_norms - self.delta_var, min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=(2, 3, 4))
        num_voxels_per_instance = torch.sum(target, dim=(2, 3, 4))
        C = target.size()[1]
        variance_term = torch.sum(embedding_variance / num_voxels_per_instance, dim=1) / C
        return variance_term

    def _compute_distance_term(self, cluster_means, C):
        if C == 1:
            return 0.0
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C
        cm_matrix1 = cluster_means.expand(shape)
        cm_matrix2 = cm_matrix1.permute(0, 2, 1, 3)
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=3)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        repulsion_dist = repulsion_dist.unsqueeze(0)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        hinged_dist = torch.sum(hinged_dist, dim=(1, 2))
        return hinged_dist / (C * (C - 1))

    def _compute_regularizer_term(self, cluster_means, C):
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        return torch.sum(norms, dim=1).div(C)

    def forward(self, input, target):
        """
        Args:
             input (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)
        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """
        C = torch.unique(target).size()[0]
        target = expand_as_one_hot(target, C)
        assert input.dim() == target.dim() == 5
        assert input.size()[2:] == target.size()[2:]
        cluster_means, embeddings_per_instance = self._compute_cluster_means(input, target)
        variance_term = self._compute_variance_term(cluster_means, embeddings_per_instance, target)
        distance_term = self._compute_distance_term(cluster_means, C)
        regularization_term = self._compute_regularizer_term(cluster_means, C)
        loss = self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term
        return torch.mean(loss)


class DiceLoss2D(nn.Module):

    def __init__(self, classes, epsilon=1e-05, sigmoid_normalization=True):
        super(DiceLoss2D, self).__init__()
        self.epsilon = epsilon
        self.classes = classes
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def flatten(self, tensor):
        return tensor.view(self.classes, -1)

    def expand_as_one_hot(self, target):
        """
        Converts label image to CxHxW, where each label gets converted to
        its corresponding one-hot vector
        :param target is of shape  (1xHxW)
        :return: 3D output tensor (CxHxW) where C is the classes
        """
        shape = target.size()
        shape = list(shape)
        shape.insert(1, self.classes)
        shape = tuple(shape)
        src = target.unsqueeze(1)
        return torch.zeros(shape).scatter_(1, src, 1).squeeze(0)

    def compute_per_channel_dice(self, input, target):
        epsilon = 1e-05
        target = self.expand_as_one_hot(target.long())
        assert input.size() == target.size(), "input' and 'target' must have the same shape" + str(input.size()) + ' and ' + str(target.size())
        input = self.flatten(input)
        target = self.flatten(target).float()
        intersect = (input * target).sum(-1)
        denominator = (input + target).sum(-1)
        return 2.0 * intersect / denominator.clamp(min=epsilon)

    def forward(self, input, target):
        input = self.normalization(input)
        per_channel_dice = self.compute_per_channel_dice(input, target)
        DSC = per_channel_dice.clone().cpu().detach().numpy()
        return torch.mean(1.0 - per_channel_dice), DSC


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'
        target = target[:, :-1, ...]
        if self.squeeze_channel:
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask
        return self.loss(input, target)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, classes=4, sigmoid_normalization=True, skip_index_after=None, epsilon=1e-06):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        assert input.size() == target.size()
        input = flatten(input)
        target = flatten(target)
        target = target.float()
        if input.size(0) == 1:
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False
        intersect = (input * target).sum(-1)
        intersect = intersect * w_l
        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        return 2 * (intersect.sum() / denominator.sum())


class PixelWiseCrossEntropyLoss(nn.Module):

    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        log_probabilities = self.log_softmax(input)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float()
        else:
            class_weights = self.class_weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)
        weights = class_weights * weights
        result = -weights * target * log_probabilities
        return result.mean()


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.
    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses3D
    """
    assert input.size() == target.size()
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-08) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-08) * stability_coeff
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


class TagsAngularLoss(torch.nn.Module):

    def __init__(self, tags_coefficients=[1.0, 0.8, 0.5], classes=4):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients
        self.classes = classes

    def forward(self, inputs, targets, weight=None):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            """
            New code here: add expand for consistency
            """
            target = expand_as_one_hot(target, self.classes)
            assert input.size() == target.size(), "'input' and 'target' must have the same shape"
            loss += alpha * square_angular_loss(input, target, weight)
        return loss


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return torch.nn.functional.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        input = torch.nn.functional.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(nominator / denominator, requires_grad=False)
        return class_weights


class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):

    def __init__(self, threshold=0, initial_weight=0.1, apply_below_threshold=True, classes=4):
        super().__init__(reduction='none')
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight
        self.classes = classes

    def forward(self, input, target):
        target = expand_as_one_hot(target, self.classes)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        l1 = super().forward(input, target)
        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold
        l1[mask] = l1[mask] * self.weight
        return l1.mean()


class BaseModel(nn.Module, ABC):
    """
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.best_loss = 1000000

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        """
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError('No checkpoint file to be restored.')
        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt_dict['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        return ckpt_dict['epoch']

    def save_checkpoint(self, directory, epoch, loss, optimizer=None, name=None):
        """
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        ckpt_dict = {'model_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None, 'epoch': epoch}
        if name is None:
            name = '{}_{}_epoch.pth'.format(os.path.basename(directory), 'last')
        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = '{}_BEST.pth'.format(os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        """
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class PEPX(nn.Module):

    def __init__(self, n_input, n_out):
        super(PEPX, self).__init__()
        """
        • First-stage Projection: 1×1 convolutions for projecting input features to a lower dimension,

        • Expansion: 1×1 convolutions for expanding features
            to a higher dimension that is different than that of the
            input features,


        • Depth-wise Representation: efficient 3×3 depthwise convolutions for learning spatial characteristics to
            minimize computational complexity while preserving
            representational capacity,

        • Second-stage Projection: 1×1 convolutions for projecting features back to a lower dimension, and

        • Extension: 1×1 convolutions that finally extend channel dimensionality to a higher dimension to produce
             the final features.

        """
        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1), nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4), kernel_size=1), nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4), kernel_size=3, groups=int(3 * n_input / 4), padding=1), nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2, kernel_size=1), nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1))

    def forward(self, x):
        return self.network(x)


class CovidNet(nn.Module):

    def __init__(self, model='large', n_classes=3):
        super(CovidNet, self).__init__()
        filters = {'pepx1_1': [64, 256], 'pepx1_2': [256, 256], 'pepx1_3': [256, 256], 'pepx2_1': [256, 512], 'pepx2_2': [512, 512], 'pepx2_3': [512, 512], 'pepx2_4': [512, 512], 'pepx3_1': [512, 1024], 'pepx3_2': [1024, 1024], 'pepx3_3': [1024, 1024], 'pepx3_4': [1024, 1024], 'pepx3_5': [1024, 1024], 'pepx3_6': [1024, 1024], 'pepx4_1': [1024, 2048], 'pepx4_2': [2048, 2048], 'pepx4_3': [2048, 2048]}
        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        for key in filters:
            if 'pool' in key:
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))
            else:
                self.add_module(key, pepx(filters[key][0], filters[key][1]))
        if model == 'large':
            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1))
            self.__forward__ = self.forward_large_net
        else:
            self.__forward__ = self.forward_small_net
        self.add_module('flatten', Flatten())
        self.add_module('fc1', nn.Linear(7 * 7 * 2048, 1024))
        self.add_module('fc2', nn.Linear(1024, 256))
        self.add_module('classifier', nn.Linear(256, n_classes))

    def forward(self, x):
        return self.__forward__(x)

    def forward_large_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out_conv1_1x1 = self.conv1_1x1(x)
        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pepx1_3(pepx12 + pepx11 + out_conv1_1x1)
        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx12 + pepx11 + pepx13 + out_conv1_1x1), 2)
        pepx21 = self.pepx2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1, 2))
        pepx22 = self.pepx2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pepx2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)
        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1), 2)
        pepx31 = self.pepx3_1(F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2) + F.max_pool2d(out_conv2_1x1, 2))
        pepx32 = self.pepx3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pepx3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)
        out_conv4_1x1 = F.max_pool2d(self.conv4_1x1(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1), 2)
        pepx41 = self.pepx4_1(F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34, 2) + F.max_pool2d(pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(out_conv3_1x1, 2))
        pepx42 = self.pepx4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pepx4_3(pepx41 + pepx42 + out_conv4_1x1)
        flattened = self.flatten(pepx41 + pepx42 + pepx43 + out_conv4_1x1)
        fc1out = F.relu(self.fc1(flattened))
        fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc2out)
        return logits

    def forward_small_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        pepx11 = self.pepx1_1(x)
        pepx12 = self.pepx1_2(pepx11)
        pepx13 = self.pepx1_3(pepx12 + pepx11)
        pepx21 = self.pepx2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2))
        pepx22 = self.pepx2_2(pepx21)
        pepx23 = self.pepx2_3(pepx22 + pepx21)
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22)
        pepx31 = self.pepx3_1(F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2))
        pepx32 = self.pepx3_2(pepx31)
        pepx33 = self.pepx3_3(pepx31 + pepx32)
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)
        pepx41 = self.pepx4_1(F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34, 2) + F.max_pool2d(pepx35, 2) + F.max_pool2d(pepx36, 2))
        pepx42 = self.pepx4_2(pepx41)
        pepx43 = self.pepx4_3(pepx41 + pepx42)
        flattened = self.flatten(pepx41 + pepx42 + pepx43)
        fc1out = F.relu(self.fc1(flattened))
        fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc2out)
        return logits


class CNN(nn.Module):

    def __init__(self, classes, model='resnet18'):
        super(CNN, self).__init__()
        if model == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Linear(512, classes)
        elif model == 'resnext50_32x4d':
            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)
        elif model == 'mobilenet_v2':
            self.cnn = models.mobilenet_v2(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)

    def forward(self, x):
        return self.cnn(x)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))


class _Upsampling(nn.Sequential):
    """
    For transpose conv
    o = output, p = padding, k = kernel_size, s = stride, d = dilation
    o = (i -1)*s - 2*p + k + output_padding = (i-1)*2 +2 = 2*i
    """

    def __init__(self, input_features, out_features):
        super(_Upsampling, self).__init__()
        self.tr_conv1_features = 128
        self.tr_conv2_features = out_features
        self.add_module('norm', nn.BatchNorm3d(input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(input_features, input_features, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('transp_conv_1', nn.ConvTranspose3d(input_features, self.tr_conv1_features, kernel_size=2, padding=0, output_padding=0, stride=2))
        self.add_module('transp_conv_2', nn.ConvTranspose3d(self.tr_conv1_features, self.tr_conv2_features, kernel_size=2, padding=0, output_padding=0, stride=2))


class DenseVoxelNet(BaseModel):
    """
    Implementation based on https://arxiv.org/abs/1708.00573
    Trainable params: 1,783,408 (roughly 1.8 mentioned in the paper)
    """

    def __init__(self, in_channels=1, classes=3):
        super(DenseVoxelNet, self).__init__()
        num_input_features = 16
        self.dense_1_out_features = 160
        self.dense_2_out_features = 304
        self.up_out_features = 64
        self.classes = classes
        self.in_channels = in_channels
        self.conv_init = nn.Conv3d(in_channels, num_input_features, kernel_size=1, stride=2, padding=0, bias=False)
        self.dense_1 = _DenseBlock(num_layers=12, num_input_features=num_input_features, bn_size=1, growth_rate=12)
        self.trans = _Transition(self.dense_1_out_features, self.dense_1_out_features)
        self.dense_2 = _DenseBlock(num_layers=12, num_input_features=self.dense_1_out_features, bn_size=1, growth_rate=12)
        self.up_block = _Upsampling(self.dense_2_out_features, self.up_out_features)
        self.conv_final = nn.Conv3d(self.up_out_features, classes, kernel_size=1, padding=0, bias=False)
        self.transpose = nn.ConvTranspose3d(self.dense_1_out_features, self.up_out_features, kernel_size=2, padding=0, output_padding=0, stride=2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.dense_1(x)
        x, t = self.trans(x)
        x = self.dense_2(x)
        x = self.up_block(x)
        y1 = self.conv_final(x)
        t = self.transpose(t)
        y2 = self.conv_final(t)
        return y1, y2

    def test(self, device='cpu'):
        a = torch.rand(1, self.in_channels, 8, 8, 8)
        ideal_out = torch.rand(1, self.classes, 8, 8, 8)
        summary(self, (self.in_channels, 8, 8, 8), device=device)
        b, c = self.forward(a)
        torchsummaryX.summary(self, a)
        assert ideal_out.shape == b.shape
        assert ideal_out.shape == c.shape
        None


class _HyperDenseLayer(nn.Sequential):

    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, num_output_channels, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _HyperDenseBlock(nn.Sequential):
    """
    Constructs a series of dense-layers based on in and out kernels list
    """

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlock, self).__init__()
        out_kernels = [1, 25, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 9
        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])
        None
        None
        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _HyperDenseBlockEarlyFusion(nn.Sequential):

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlockEarlyFusion, self).__init__()
        out_kernels = [1, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 8
        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])
        None
        None
        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class SinglePathDenseNet(BaseModel):

    def __init__(self, in_channels, classes=4, drop_rate=0.1, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.num_classes = classes
        self.input_channels = in_channels
        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 52:
                total_conv_channels = 477
            elif in_channels == 3:
                total_conv_channels = 426
            else:
                total_conv_channels = 503
        else:
            block = _HyperDenseBlock(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 2:
                total_conv_channels = 452
            else:
                total_conv_channels = 451
        self.features.add_module('denseblock1', block)
        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels, 400, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.add_module('drop_1', nn.Dropout(p=0.5))
        self.features.add_module('conv1x1_2', nn.Conv3d(400, 200, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.add_module('drop_2', nn.Dropout(p=0.5))
        self.features.add_module('conv1x1_3', nn.Conv3d(200, 150, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.add_module('drop_3', nn.Dropout(p=0.5))
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        features = self.features(x)
        if self.return_logits:
            out = self.classifier(features)
            return out
        else:
            return features

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.input_channels, 12, 12, 12), device=device)
        None


class DualPathDenseNet(BaseModel):

    def __init__(self, in_channels, classes=4, drop_rate=0, fusion='concat'):
        """
        2-stream and 3-stream implementation with late fusion
        :param in_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualPathDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes
        self.fusion = fusion
        if self.fusion == 'concat':
            in_classifier_channels = self.input_channels * 150
        else:
            in_classifier_channels = 150
        if self.input_channels == 2:
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes, return_logits=False, early_fusion=True)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes, return_logits=False, early_fusion=True)
        if self.input_channels == 3:
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes, return_logits=False)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes, return_logits=False)
            self.stream_3 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes, return_logits=False)
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(in_classifier_channels, classes, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            None
            return None
        elif self.input_channels == 2:
            in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
            in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
            output_features_t1 = self.stream_1(in_stream_1)
            output_features_t2 = self.stream_2(in_stream_2)
            if self.fusion == 'concat':
                concat_features = torch.cat((output_features_t1, output_features_t2), dim=1)
                return self.classifier(concat_features)
            else:
                features = output_features_t1 + output_features_t2
                return self.classifier(features)
        elif self.input_channels == 3:
            in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
            in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
            in_stream_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
            output_features_t1 = self.stream_1(in_stream_1)
            output_features_t2 = self.stream_2(in_stream_2)
            output_features_t3 = self.stream_3(in_stream_3)
            if self.fusion == 'concat':
                concat_features = torch.cat((output_features_t1, output_features_t2, output_features_t3), dim=1)
                return self.classifier(concat_features)
            else:
                features = output_features_t1 + output_features_t2 + output_features_t3
                return self.classifier(features)

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.input_channels, 12, 12, 12), device=device)
        torchsummaryX.summary(self, input_tensor)
        None


class DualSingleDenseNet(BaseModel):
    """
    2-stream and 3-stream implementation with early fusion
    dual-single-densenet OR Disentangled modalities with early fusion in the paper
    """

    def __init__(self, in_channels, classes=4, drop_rate=0.5):
        """

        :param input_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualSingleDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes
        if self.input_channels == 2:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            single_path_channels = 52
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate, classes=classes, return_logits=True, early_fusion=True)
            self.classifier = nn.Sequential()
        if self.input_channels == 3:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_3 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            single_path_channels = 78
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate, classes=classes, return_logits=True, early_fusion=True)

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            None
            return None
        elif self.input_channels == 2:
            in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
            in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
            y1 = self.early_conv_1(in_1)
            y2 = self.early_conv_1(in_2)
            None
            None
            in_stream = torch.cat((y1, y2), dim=1)
            logits = self.stream_1(in_stream)
            return logits
        elif self.input_channels == 3:
            in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
            in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
            in_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
            y1 = self.early_conv_1(in_1)
            y2 = self.early_conv_2(in_2)
            y3 = self.early_conv_3(in_3)
            in_stream = torch.cat((y1, y2, y3), dim=1)
            logits = self.stream_1(in_stream)
            return logits

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.input_channels, 12, 12, 12), device=device)
        None


class ConvInit(nn.Module):

    def __init__(self, in_channels):
        super(ConvInit, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        bn1 = torch.nn.BatchNorm3d(self.num_features)
        relu1 = nn.ReLU()
        self.norm = nn.Sequential(bn1, relu1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.norm(y1)
        return y1, y2


class ConvRed(nn.Module):

    def __init__(self, in_channels):
        super(ConvRed, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        self.conv_red = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_red(x)


class DilatedConv2(nn.Module):

    def __init__(self, in_channels):
        super(DilatedConv2, self).__init__()
        self.num_features = 32
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=2, dilation=2)
        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class DilatedConv4(nn.Module):

    def __init__(self, in_channels):
        super(DilatedConv4, self).__init__()
        self.num_features = 64
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=4, dilation=4)
        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class Conv1x1x1(nn.Module):

    def __init__(self, in_channels, classes):
        super(Conv1x1x1, self).__init__()
        self.num_features = classes
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=1)
        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class HighResNet3D(BaseModel):

    def __init__(self, in_channels=1, classes=4, shortcut_type='A', dropout_layer=True):
        super(HighResNet3D, self).__init__()
        self.in_channels = in_channels
        self.shortcut_type = shortcut_type
        self.classes = classes
        self.init_channels = 16
        self.red_channels = 16
        self.dil2_channels = 32
        self.dil4_channels = 64
        self.conv_out_channels = 80
        if self.shortcut_type == 'B':
            self.res_pad_1 = Conv1x1x1(self.red_channels, self.dil2_channels)
            self.res_pad_2 = Conv1x1x1(self.dil2_channels, self.dil4_channels)
        self.conv_init = ConvInit(in_channels)
        self.red_blocks1 = self.create_red(self.init_channels)
        self.red_blocks2 = self.create_red(self.red_channels)
        self.red_blocks3 = self.create_red(self.red_channels)
        self.dil2block1 = self.create_dil2(self.red_channels)
        self.dil2block2 = self.create_dil2(self.dil2_channels)
        self.dil2block3 = self.create_dil2(self.dil2_channels)
        self.dil4block1 = self.create_dil4(self.dil2_channels)
        self.dil4block2 = self.create_dil4(self.dil4_channels)
        self.dil4block3 = self.create_dil4(self.dil4_channels)
        if dropout_layer:
            conv_out = nn.Conv3d(self.dil4_channels, self.conv_out_channels, kernel_size=1)
            drop3d = nn.Dropout3d()
            conv1x1x1 = Conv1x1x1(self.conv_out_channels, self.classes)
            self.conv_out = nn.Sequential(conv_out, drop3d, conv1x1x1)
        else:
            self.conv_out = Conv1x1x1(self.dil4_channels, self.classes)

    def shortcut_pad(self, x, desired_channels):
        if self.shortcut_type == 'A':
            batch_size, channels, dim0, dim1, dim2 = x.shape
            extra_channels = desired_channels - channels
            zero_channels = int(extra_channels / 2)
            zeros_half = x.new_zeros(batch_size, zero_channels, dim0, dim1, dim2)
            y = torch.cat((zeros_half, x, zeros_half), dim=1)
        elif self.shortcut_type == 'B':
            if desired_channels == self.dil2_channels:
                y = self.res_pad_1(x)
            elif desired_channels == self.dil4_channels:
                y = self.res_pad_2(x)
        return y

    def create_red(self, in_channels):
        conv_red_1 = ConvRed(in_channels)
        conv_red_2 = ConvRed(self.red_channels)
        return nn.Sequential(conv_red_1, conv_red_2)

    def create_dil2(self, in_channels):
        conv_dil2_1 = DilatedConv2(in_channels)
        conv_dil2_2 = DilatedConv2(self.dil2_channels)
        return nn.Sequential(conv_dil2_1, conv_dil2_2)

    def create_dil4(self, in_channels):
        conv_dil4_1 = DilatedConv4(in_channels)
        conv_dil4_2 = DilatedConv4(self.dil4_channels)
        return nn.Sequential(conv_dil4_1, conv_dil4_2)

    def red_forward(self, x):
        x, x_res = self.conv_init(x)
        x_red_1 = self.red_blocks1(x)
        x_red_2 = self.red_blocks2(x_red_1 + x_res)
        x_red_3 = self.red_blocks3(x_red_2 + x_red_1)
        return x_red_3, x_red_2

    def dilation2(self, x_red_3, x_red_2):
        x_dil2_1 = self.dil2block1(x_red_3 + x_red_2)
        x_red_padded = self.shortcut_pad(x_red_3, self.dil2_channels)
        x_dil2_2 = self.dil2block2(x_dil2_1 + x_red_padded)
        x_dil2_3 = self.dil2block3(x_dil2_2 + x_dil2_1)
        return x_dil2_3, x_dil2_2

    def dilation4(self, x_dil2_3, x_dil2_2):
        x_dil4_1 = self.dil4block1(x_dil2_3 + x_dil2_2)
        x_dil2_padded = self.shortcut_pad(x_dil2_3, self.dil4_channels)
        x_dil4_2 = self.dil4block2(x_dil4_1 + x_dil2_padded)
        x_dil4_3 = self.dil4block3(x_dil4_2 + x_dil4_1)
        return x_dil4_3 + x_dil4_2

    def forward(self, x):
        x_red_3, x_red_2 = self.red_forward(x)
        x_dil2_3, x_dil2_2 = self.dilation2(x_red_3, x_red_2)
        x_dil4 = self.dilation4(x_dil2_3, x_dil2_2)
        y = self.conv_out(x_dil4)
        return y

    def test(self):
        x = torch.rand(1, self.in_channels, 32, 32, 32)
        pred = self.forward(x)
        target = torch.rand(1, self.classes, 32, 32, 32)
        assert target.shape == pred.shape
        None


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, BN=False, ws=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            layers.append(activ(num_parameters=1))
        else:
            layers.append(activ)
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)


class ResidualConv(nn.Module):

    def __init__(self, nin, nout, bias=False, BN=False, ws=False, activ=nn.LeakyReLU(0.2)):
        super(ResidualConv, self).__init__()
        convs = [conv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ), conv(nout, nout, bias=bias, BN=BN, ws=ws, activ=None)]
        self.convs = nn.Sequential(*convs)
        res = []
        if nin != nout:
            res.append(conv(nin, nout, kernel_size=1, padding=0, bias=False, BN=BN, ws=ws, activ=None))
        self.res = nn.Sequential(*res)
        activation = []
        if activ is not None:
            if activ == nn.PReLU:
                activation.append(activ(num_parameters=1))
            else:
                activation.append(activ)
        self.activation = nn.Sequential(*activation)

    def forward(self, input):
        out = self.convs(input)
        return self.activation(out + self.res(input))


def convBlock(nin, nout, kernel_size=3, batchNorm=False, layer=nn.Conv3d, bias=True, dropout_rate=0.0, dilation=1):
    if batchNorm == False:
        return nn.Sequential(nn.PReLU(), nn.Dropout(p=dropout_rate), layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation))
    else:
        return nn.Sequential(nn.BatchNorm3d(nin), nn.PReLU(), nn.Dropout(p=dropout_rate), layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation))


def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    croppBorders = int(diff / 2)
    return tensorToCrop[:, :, croppBorders:org_shape[2] - croppBorders, croppBorders:org_shape[3] - croppBorders, croppBorders:org_shape[4] - croppBorders]


class HyperDenseNet_2Mod(BaseModel):

    def __init__(self, in_channels=2, classes=4):
        super(HyperDenseNet_2Mod, self).__init__()
        self.num_classes = classes
        assert in_channels == 2, 'input channels must be two for this architecture'
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(50, 25, batchNorm=True)
        self.conv3_Top = convBlock(100, 25, batchNorm=True)
        self.conv4_Top = convBlock(150, 50, batchNorm=True)
        self.conv5_Top = convBlock(250, 50, batchNorm=True)
        self.conv6_Top = convBlock(350, 50, batchNorm=True)
        self.conv7_Top = convBlock(450, 75, batchNorm=True)
        self.conv8_Top = convBlock(600, 75, batchNorm=True)
        self.conv9_Top = convBlock(750, 75, batchNorm=True)
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(50, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(100, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(150, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(250, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(350, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(450, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(600, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(750, 75, batchNorm=True)
        self.fully_1 = nn.Conv3d(1800, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, classes, kernel_size=1)

    def forward(self, input):
        None
        y1t = self.conv1_Top(input[:, 0:1, :, :, :])
        y1b = self.conv1_Bottom(input[:, 1:2, :, :, :])
        y2t_i = torch.cat((y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t), dim=1)
        y2t_o = self.conv2_Top(y2t_i)
        y2b_o = self.conv2_Bottom(y2b_i)
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o), dim=1)
        y3t_o = self.conv3_Top(y3t_i)
        y3b_o = self.conv3_Bottom(y3b_i)
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o), dim=1)
        y4t_o = self.conv4_Top(y4t_i)
        y4b_o = self.conv4_Bottom(y4b_i)
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o), dim=1)
        y5t_o = self.conv5_Top(y5t_i)
        y5b_o = self.conv5_Bottom(y5b_i)
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)
        y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5b_o), dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o), dim=1)
        y6t_o = self.conv6_Top(y6t_i)
        y6b_o = self.conv6_Bottom(y6b_i)
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)
        y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6b_o), dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o), dim=1)
        y7t_o = self.conv7_Top(y7t_i)
        y7b_o = self.conv7_Bottom(y7b_i)
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)
        y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7b_o), dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o), dim=1)
        y8t_o = self.conv8_Top(y8t_i)
        y8b_o = self.conv8_Bottom(y8b_i)
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)
        y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8b_o), dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o), dim=1)
        y9t_o = self.conv9_Top(y9t_i)
        y9b_o = self.conv9_Bottom(y9b_i)
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)
        outputPath_top = torch.cat((y9t_i_cropped, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o), dim=1)
        inputFully = torch.cat((outputPath_top, outputPath_bottom), dim=1)
        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)
        return self.final(y)

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, 2, 22, 22, 22)
        ideal_out = torch.rand(1, self.num_classes, 22, 22, 22)
        out = self.forward(input_tensor)
        None


class HyperDenseNet(BaseModel):

    def __init__(self, in_channels=3, classes=4):
        super(HyperDenseNet, self).__init__()
        assert in_channels == 3, 'HyperDensenet supports 3 in_channels. For 2 in_channels use HyperDenseNet_2Mod '
        self.num_classes = classes
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(75, 25, batchNorm=True)
        self.conv3_Top = convBlock(150, 25, batchNorm=True)
        self.conv4_Top = convBlock(225, 50, batchNorm=True)
        self.conv5_Top = convBlock(375, 50, batchNorm=True)
        self.conv6_Top = convBlock(525, 50, batchNorm=True)
        self.conv7_Top = convBlock(675, 75, batchNorm=True)
        self.conv8_Top = convBlock(900, 75, batchNorm=True)
        self.conv9_Top = convBlock(1125, 75, batchNorm=True)
        self.conv1_Middle = convBlock(1, 25)
        self.conv2_Middle = convBlock(75, 25, batchNorm=True)
        self.conv3_Middle = convBlock(150, 25, batchNorm=True)
        self.conv4_Middle = convBlock(225, 50, batchNorm=True)
        self.conv5_Middle = convBlock(375, 50, batchNorm=True)
        self.conv6_Middle = convBlock(525, 50, batchNorm=True)
        self.conv7_Middle = convBlock(675, 75, batchNorm=True)
        self.conv8_Middle = convBlock(900, 75, batchNorm=True)
        self.conv9_Middle = convBlock(1125, 75, batchNorm=True)
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(75, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(150, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(225, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(375, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(525, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(675, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(900, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(1125, 75, batchNorm=True)
        self.fully_1 = nn.Conv3d(4050, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, classes, kernel_size=1)

    def forward(self, input):
        y1t = self.conv1_Top(input[:, 0:1, :, :, :])
        y1m = self.conv1_Middle(input[:, 1:2, :, :, :])
        y1b = self.conv1_Bottom(input[:, 2:3, :, :, :])
        y2t_i = torch.cat((y1t, y1m, y1b), dim=1)
        y2m_i = torch.cat((y1m, y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t, y1m), dim=1)
        y2t_o = self.conv2_Top(y2t_i)
        y2m_o = self.conv2_Middle(y2m_i)
        y2b_o = self.conv2_Bottom(y2b_i)
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2m_i_cropped = croppCenter(y2m_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2m_o, y2b_o), dim=1)
        y3m_i = torch.cat((y2m_i_cropped, y2m_o, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o, y2m_o), dim=1)
        y3t_o = self.conv3_Top(y3t_i)
        y3m_o = self.conv3_Middle(y3m_i)
        y3b_o = self.conv3_Bottom(y3b_i)
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3m_i_cropped = croppCenter(y3m_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3m_o, y3b_o), dim=1)
        y4m_i = torch.cat((y3m_i_cropped, y3m_o, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o, y3m_o), dim=1)
        y4t_o = self.conv4_Top(y4t_i)
        y4m_o = self.conv4_Middle(y4m_i)
        y4b_o = self.conv4_Bottom(y4b_i)
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4m_i_cropped = croppCenter(y4m_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4m_o, y4b_o), dim=1)
        y5m_i = torch.cat((y4m_i_cropped, y4m_o, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o, y4m_o), dim=1)
        y5t_o = self.conv5_Top(y5t_i)
        y5m_o = self.conv5_Middle(y5m_i)
        y5b_o = self.conv5_Bottom(y5b_i)
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5m_i_cropped = croppCenter(y5m_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)
        y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5m_o, y5b_o), dim=1)
        y6m_i = torch.cat((y5m_i_cropped, y5m_o, y5t_o, y5b_o), dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o, y5m_o), dim=1)
        y6t_o = self.conv6_Top(y6t_i)
        y6m_o = self.conv6_Middle(y6m_i)
        y6b_o = self.conv6_Bottom(y6b_i)
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6m_i_cropped = croppCenter(y6m_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)
        y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6m_o, y6b_o), dim=1)
        y7m_i = torch.cat((y6m_i_cropped, y6m_o, y6t_o, y6b_o), dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o, y6m_o), dim=1)
        y7t_o = self.conv7_Top(y7t_i)
        y7m_o = self.conv7_Middle(y7m_i)
        y7b_o = self.conv7_Bottom(y7b_i)
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7m_i_cropped = croppCenter(y7m_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)
        y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7m_o, y7b_o), dim=1)
        y8m_i = torch.cat((y7m_i_cropped, y7m_o, y7t_o, y7b_o), dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o, y7m_o), dim=1)
        y8t_o = self.conv8_Top(y8t_i)
        y8m_o = self.conv8_Middle(y8m_i)
        y8b_o = self.conv8_Bottom(y8b_i)
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8m_i_cropped = croppCenter(y8m_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)
        y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8m_o, y8b_o), dim=1)
        y9m_i = torch.cat((y8m_i_cropped, y8m_o, y8t_o, y8b_o), dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o, y8m_o), dim=1)
        y9t_o = self.conv9_Top(y9t_i)
        y9m_o = self.conv9_Middle(y9m_i)
        y9b_o = self.conv9_Bottom(y9b_i)
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9m_i_cropped = croppCenter(y9m_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)
        outputPath_top = torch.cat((y9t_i_cropped, y9t_o, y9m_o, y9b_o), dim=1)
        outputPath_middle = torch.cat((y9m_i_cropped, y9m_o, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o, y9m_o), dim=1)
        inputFully = torch.cat((outputPath_top, outputPath_middle, outputPath_bottom), dim=1)
        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)
        return self.final(y)

    def test(self, device='cpu'):
        device = torch.device(device)
        input_tensor = torch.rand(1, 3, 20, 20, 20)
        ideal_out = torch.rand(1, self.num_classes, 20, 20, 20)
        out = self.forward(input_tensor)
        summary(self, (3, 16, 16, 16))
        None


def find_padding(dilation, kernel):
    """
    Dynamically computes padding to keep input conv size equal to the output
    for stride = 1
    :return:
    """
    return int(((kernel - 1) * (dilation - 1) + (kernel - 1)) / 2.0)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    kernel_size = 3
    if dilation > 1:
        padding = find_padding(dilation, kernel_size)
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class TranspConvNet(nn.Module):
    """
    (segmentation)we transfer encoder part from Med3D as the feature extraction part and 
    then segmented lung in whole body followed by three groups of 3D decoder layers.
    The first set of decoder layers is composed of a transposed
    convolution layer with a kernel size of(3,3,3)and a channel number of 256 
    (which isused to amplify twice the feature map), and the convolutional layer with(3,3,3)kernel
    size and 128 channels.
    """

    def __init__(self, in_channels, classes):
        super().__init__()
        conv_channels = 128
        transp_channels = 256
        transp_conv_1 = nn.ConvTranspose3d(in_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_1 = nn.BatchNorm3d(transp_channels)
        relu_1 = nn.ReLU(inplace=True)
        self.transp_1 = nn.Sequential(transp_conv_1, batch_norm_1, relu_1)
        transp_conv_2 = nn.ConvTranspose3d(transp_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_2 = nn.BatchNorm3d(transp_channels)
        relu_2 = nn.ReLU(inplace=True)
        self.transp_2 = nn.Sequential(transp_conv_2, batch_norm_2, relu_2)
        transp_conv_3 = nn.ConvTranspose3d(transp_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_3 = nn.BatchNorm3d(transp_channels)
        relu_3 = nn.ReLU(inplace=True)
        self.transp_3 = nn.Sequential(transp_conv_3, batch_norm_3, relu_3)
        conv1 = conv3x3x3(transp_channels, conv_channels, stride=1, padding=1)
        batch_norm_2 = nn.BatchNorm3d(conv_channels)
        relu_2 = nn.ReLU(inplace=True)
        self.conv_1 = nn.Sequential(conv1, batch_norm_2, relu_2)
        self.conv_final = conv1x1x1(conv_channels, classes, stride=1)

    def forward(self, x):
        x = self.transp_1(x)
        x = self.transp_2(x)
        x = self.transp_3(x)
        x = self.conv_1(x)
        y = self.conv_final(x)
        return y


class ResNetMed3D(BaseModel):

    def __init__(self, in_channels=3, classes=10, block=BasicBlock, layers=[1, 1, 1, 1], block_inplanes=[64, 128, 256, 512], no_max_pool=False, shortcut_type='B', widen_factor=1.0):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels, self.in_planes, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=1, dilation=4)
        self.segm = TranspConvNet(in_channels=512 * block.expansion, classes=classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.FloatTensor):
            zero_pads = zero_pads
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, dilation=dilation, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        None
        x = self.segm(x)
        return x

    def test(self):
        a = torch.rand(1, self.in_channels, 16, 16, 16)
        y = self.forward(a)
        target = torch.rand(1, self.classes, 16, 16, 16)
        assert a.shape == y.shape


class GreenBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32, norm='group'):
        super(GreenBlock, self).__init__()
        if norm == 'batch':
            norm_1 = nn.BatchNorm3d(num_features=in_channels)
            norm_2 = nn.BatchNorm3d(num_features=in_channels)
        elif norm == 'group':
            norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            norm_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.layer_1 = nn.Sequential(norm_1, nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=1, padding=1), norm_2, nn.ReLU())
        self.conv_3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        y = self.conv_3(x)
        y = y + x
        return y


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class BlueBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32):
        super(BlueBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpBlock1(nn.Module):
    """
    TODO fix transpose conv to double spatial dim
    """

    def __init__(self, in_channels, out_channels):
        super(UpBlock1, self).__init__()
        self.transp_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=2, padding=1)

    def forward(self, x):
        return self.transp_conv(x)


class UpBlock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=1)
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up_sample_1(self.conv_1(x))


class ResNetEncoder(nn.Module):

    def __init__(self, in_channels, start_channels=32):
        super(ResNetEncoder, self).__init__()
        self.start_channels = start_channels
        self.down_channels_1 = 2 * self.start_channels
        self.down_channels_2 = 2 * self.down_channels_1
        self.down_channels_3 = 2 * self.down_channels_2
        self.blue_1 = BlueBlock(in_channels=in_channels, out_channels=self.start_channels)
        self.drop = nn.Dropout3d(0.2)
        self.green_1 = GreenBlock(in_channels=self.start_channels)
        self.down_1 = DownBlock(in_channels=self.start_channels, out_channels=self.down_channels_1)
        self.green_2_1 = GreenBlock(in_channels=self.down_channels_1)
        self.green_2_2 = GreenBlock(in_channels=self.down_channels_1)
        self.down_2 = DownBlock(in_channels=self.down_channels_1, out_channels=self.down_channels_2)
        self.green_3_1 = GreenBlock(in_channels=self.down_channels_2)
        self.green_3_2 = GreenBlock(in_channels=self.down_channels_2)
        self.down_3 = DownBlock(in_channels=self.down_channels_2, out_channels=self.down_channels_3)
        self.green_4_1 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_2 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_3 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_4 = GreenBlock(in_channels=self.down_channels_3)

    def forward(self, x):
        x = self.blue_1(x)
        x = self.drop(x)
        x1 = self.green_1(x)
        x = self.down_1(x1)
        x = self.green_2_1(x)
        x2 = self.green_2_2(x)
        x = self.down_2(x2)
        x = self.green_3_1(x)
        x3 = self.green_3_2(x)
        x = self.down_3(x3)
        x = self.green_4_1(x)
        x = self.green_4_2(x)
        x = self.green_4_3(x)
        x4 = self.green_4_4(x)
        return x1, x2, x3, x4


class Decoder(nn.Module):

    def __init__(self, in_channels=256, classes=4):
        super(Decoder, self).__init__()
        out_up_1_channels = int(in_channels / 2)
        out_up_2_channels = int(out_up_1_channels / 2)
        out_up_3_channels = int(out_up_2_channels / 2)
        self.up_1 = UpBlock2(in_channels=in_channels, out_channels=out_up_1_channels)
        self.green_1 = GreenBlock(in_channels=out_up_1_channels)
        self.up_2 = UpBlock2(in_channels=out_up_1_channels, out_channels=out_up_2_channels)
        self.green_2 = GreenBlock(in_channels=out_up_2_channels)
        self.up_3 = UpBlock2(in_channels=out_up_2_channels, out_channels=out_up_3_channels)
        self.green_3 = GreenBlock(in_channels=out_up_3_channels)
        self.blue = BlueBlock(in_channels=out_up_3_channels, out_channels=classes)

    def forward(self, x1, x2, x3, x4):
        x = self.up_1(x4)
        x = self.green_1(x + x3)
        x = self.up_2(x)
        x = self.green_2(x + x2)
        x = self.up_3(x)
        x = self.green_3(x + x1)
        y = self.blue(x)
        return y


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class VAE(nn.Module):

    def __init__(self, in_channels=256, in_dim=(10, 10, 10), out_dim=(2, 64, 64, 64)):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = out_dim[0]
        self.encoder_channels = 16
        self.split_dim = int(self.in_channels / 2)
        self.reshape_dim = int(self.out_dim[1] / self.encoder_channels), int(self.out_dim[2] / self.encoder_channels), int(self.out_dim[3] / self.encoder_channels)
        self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) * (in_dim[1] / 2) * (in_dim[2] / 2))
        self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2]
        channels_vup2 = int(self.in_channels / 2)
        channels_vup1 = int(channels_vup2 / 2)
        channels_vup0 = int(channels_vup1 / 2)
        group_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        relu_1 = nn.ReLU()
        conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=self.encoder_channels, stride=2, kernel_size=(3, 3, 3), padding=1)
        self.VD = nn.Sequential(group_1, relu_1, conv_1)
        self.linear_1 = nn.Linear(self.linear_in_dim, in_channels)
        self.linear_vu = nn.Linear(channels_vup2, self.linear_vu_dim)
        relu_vu = nn.ReLU()
        VUup_block = UpBlock2(in_channels=self.encoder_channels, out_channels=self.in_channels)
        self.VU = nn.Sequential(relu_vu, VUup_block)
        self.Vup2 = UpBlock2(in_channels, channels_vup2)
        self.Vblock2 = GreenBlock(channels_vup2)
        self.Vup1 = UpBlock2(channels_vup2, channels_vup1)
        self.Vblock1 = GreenBlock(channels_vup1)
        self.Vup0 = UpBlock2(channels_vup1, channels_vup0)
        self.Vblock0 = GreenBlock(channels_vup0)
        self.Vend = BlueBlock(channels_vup0, self.modalities)

    def forward(self, x):
        x = self.VD(x)
        x = x.view(-1, self.linear_in_dim)
        x = self.linear_1(x)
        mu = x[:, :self.split_dim]
        logvar = x[:, self.split_dim:]
        y = reparametrize(mu, logvar)
        y = self.linear_vu(y)
        y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.reshape_dim[1], self.reshape_dim[2])
        y = self.VU(y)
        y = self.Vup2(y)
        y = self.Vblock2(y)
        y = self.Vup1(y)
        y = self.Vblock1(y)
        y = self.Vup0(y)
        y = self.Vblock0(y)
        dec = self.Vend(y)
        return dec, mu, logvar


class ResNet3dVAE(BaseModel):

    def __init__(self, in_channels=2, classes=4, max_conv_channels=256, dim=(64, 64, 64)):
        super(ResNet3dVAE, self).__init__()
        self.dim = dim
        vae_in_dim = int(dim[0] >> 3), int(dim[1] >> 3), int(dim[2] >> 3)
        vae_out_dim = in_channels, dim[0], dim[1], dim[2]
        self.classes = classes
        self.modalities = in_channels
        start_channels = 32
        self.encoder = ResNetEncoder(in_channels=in_channels, start_channels=start_channels)
        self.decoder = Decoder(in_channels=max_conv_channels, classes=classes)
        self.vae = VAE(in_channels=max_conv_channels, in_dim=vae_in_dim, out_dim=vae_out_dim)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        y = self.decoder(x1, x2, x3, x4)
        vae_out, mu, logvar = self.vae(x4)
        return y, vae_out, mu, logvar

    def test(self):
        inp = torch.rand(1, self.modalities, self.dim[0], self.dim[1], self.dim[2])
        ideal = torch.rand(1, self.classes, self.dim[0], self.dim[1], self.dim[2])
        y, vae_out, mu, logvar = self.forward(inp)
        assert vae_out.shape == inp.shape, vae_out.shape
        assert y.shape == ideal.shape
        assert mu.shape == logvar.shape
        None


class SkipDenseNet3D(BaseModel):
    """Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Based on the implementation of https://github.com/tbuikr/3D-SkipDenseSeg
    Paper here : https://arxiv.org/pdf/1709.03199.pdf

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        classes (int) - number of classification classes
    """

    def __init__(self, in_channels=2, classes=4, growth_rate=16, block_config=(4, 4, 4, 4), num_init_features=32, drop_rate=0.1, bn_size=4):
        super(SkipDenseNet3D, self).__init__()
        self.num_classes = classes
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ('norm0', nn.BatchNorm3d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ('norm1', nn.BatchNorm3d(num_init_features)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        self.features_bn = nn.Sequential(OrderedDict([('norm2', nn.BatchNorm3d(num_init_features)), ('relu2', nn.ReLU(inplace=True))]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0, bias=False)
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            up_block = nn.ConvTranspose3d(num_features, classes, kernel_size=2 ** (i + 1) + 2, stride=2 ** (i + 1), padding=1, groups=classes, bias=False)
            self.upsampling_blocks.append(up_block)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                num_features = num_features // 2
        self.bn_class = nn.BatchNorm3d(classes * 4 + num_init_features)
        self.conv_class = nn.Conv3d(classes * 4 + num_init_features, classes, kernel_size=1, padding=0)
        self.relu_last = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features_bn)
        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)
        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)
        out = self.dense_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)
        out = self.dense_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)
        out = torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)
        out = self.conv_class(self.relu_last(self.bn_class(out)))
        return out

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.num_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (2, 32, 32, 32), device=device)
        torchsummaryX.summary(self, input_tensor)
        None


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(BaseModel):

    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def test(self, device='cpu'):
        device = torch.device(device)
        input_tensor = torch.rand(1, self.n_channels, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.n_channels, 32, 32, 32), device=device)
        None


class UNet3D(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 8)
        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 4)
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 2)
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter)
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def forward(self, x):
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        return seg_layer

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (2, 32, 32, 32), device='cpu')
        None


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class InputTransition(nn.Module):

    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(self.num_features)
        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


def passthrough(x, **kwargs):
    return x


class DownTransition(nn.Module):

    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)
        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(BaseModel):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=4):
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.in_channels, 32, 32, 32), device=device)
        None


class VNetLight(BaseModel):
    """
    A lighter version of Vnet that skips down_tr256 and up_tr256 in oreder to reduce time and space complexity
    """

    def __init__(self, elu=True, in_channels=1, classes=4):
        super(VNetLight, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.in_tr = InputTransition(in_channels, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self, (self.in_channels, 32, 32, 32), device=device)
        None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BlueBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNN,
     lambda: ([], {'classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1x1x1,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvInit,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvRed,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DilatedConv2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (DilatedConv4,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownTransition,
     lambda: ([], {'inChans': 4, 'nConvs': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DualPathDenseNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DualSingleDenseNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (HighResNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (InConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputTransition,
     lambda: ([], {'in_channels': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LUConv,
     lambda: ([], {'nchan': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (OutConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutputTransition,
     lambda: ([], {'in_channels': 4, 'classes': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (PEPX,
     lambda: ([], {'n_input': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet3dVAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64, 64])], {}),
     False),
    (ResNetMed3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     False),
    (ResidualConv,
     lambda: ([], {'nin': 4, 'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipDenseNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64, 64])], {}),
     False),
    (SkipLastTargetChannelWrapper,
     lambda: ([], {'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TranspConvNet,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Unet,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (UpBlock1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpBlock2,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (VNetLight,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (WeightedCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_HyperDenseBlock,
     lambda: ([], {'num_input_features': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_HyperDenseBlockEarlyFusion,
     lambda: ([], {'num_input_features': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_HyperDenseLayer,
     lambda: ([], {'num_input_features': 4, 'num_output_channels': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_MaskingLossWrapper,
     lambda: ([], {'loss': MSELoss(), 'ignore_index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (_Upsampling,
     lambda: ([], {'input_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_black0017_MedicalZooPytorch(_paritybench_base):
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

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

