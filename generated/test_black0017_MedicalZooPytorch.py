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

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch


from torch import nn as nn


from torch.nn import MSELoss


from torch.nn import SmoothL1Loss


from torch.nn import L1Loss


from abc import ABC


from abc import abstractmethod


import torch.nn.functional as F


from functools import partial


from collections import OrderedDict


from numpy import inf


import math


import numpy as np


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
    assert input.size() == target.size(
        ), "'input' and 'target' must have the same shape"
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4
    input = input.unsqueeze(1)
    shape = list(input.size())
    shape[1] = C
    if ignore_index is not None:
        mask = input.expand(shape) == ignore_index
        input = input.clone()
        input[input == ignore_index] = 0
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        result[mask] = ignore_index
        return result
    else:
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


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
        return target[:, 0:index, (...)]

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = expand_as_one_hot(target.long(), self.classes)
        assert input.dim() == target.dim(
            ) == 5, "'input' and 'target' have different number of dims"
        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            None
        assert input.size() == target.size(
            ), "'input' and 'target' must have the same shape"
        input = self.normalization(input)
        per_channel_dice = self.dice(input, target, weight=self.weight)
        loss = 1.0 - torch.mean(per_channel_dice)
        per_channel_dice = per_channel_dice.detach().cpu().numpy()
        return loss, per_channel_dice


class ContrastiveLoss(torch.nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm='fro', alpha=1.0,
        beta=1.0, gamma=0.001):
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
        num_voxels_per_instance = torch.sum(target_copy, dim=(3, 4, 5),
            keepdim=True)
        mean_embeddings = num / num_voxels_per_instance
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance,
        target):
        embedding_norms = torch.norm(embeddings_per_instance -
            cluster_means, self.norm, dim=2)
        embedding_norms = embedding_norms * target
        embedding_variance = torch.clamp(embedding_norms - self.delta_var,
            min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=(2, 3, 4))
        num_voxels_per_instance = torch.sum(target, dim=(2, 3, 4))
        C = target.size()[1]
        variance_term = torch.sum(embedding_variance /
            num_voxels_per_instance, dim=1) / C
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
        cluster_means, embeddings_per_instance = self._compute_cluster_means(
            input, target)
        variance_term = self._compute_variance_term(cluster_means,
            embeddings_per_instance, target)
        distance_term = self._compute_distance_term(cluster_means, C)
        regularization_term = self._compute_regularizer_term(cluster_means, C)
        loss = (self.alpha * variance_term + self.beta * distance_term + 
            self.gamma * regularization_term)
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
        assert input.size() == target.size(
            ), "input' and 'target' must have the same shape" + str(input.
            size()) + ' and ' + str(target.size())
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
        assert target.size(1
            ) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'
        target = target[:, :-1, (...)]
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


class PixelWiseCrossEntropyLoss(nn.Module):

    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        log_probabilities = self.log_softmax(input)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=
            self.ignore_index)
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
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-08
        ) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-08
        ) * stability_coeff
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
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients
            ):
            """
            New code here: add expand for consistency
            """
            target = expand_as_one_hot(target, self.classes)
            assert input.size() == target.size(
                ), "'input' and 'target' must have the same shape"
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
        return torch.nn.functional.cross_entropy(input, target, weight=
            weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        input = torch.nn.functional.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = torch.autograd.Variable(nominator / denominator,
            requires_grad=False)
        return class_weights


class WeightedSmoothL1Loss(torch.nn.SmoothL1Loss):

    def __init__(self, threshold=0, initial_weight=0.1,
        apply_below_threshold=True, classes=4):
        super().__init__(reduction='none')
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight
        self.classes = classes

    def forward(self, input, target):
        target = expand_as_one_hot(target, self.classes)
        assert input.size() == target.size(
            ), "'input' and 'target' must have the same shape"
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
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage,
                loc: storage)
        self.load_state_dict(ckpt_dict['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        return ckpt_dict['epoch']

    def save_checkpoint(self, directory, epoch, loss, optimizer=None, name=None
        ):
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
        ckpt_dict = {'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not
            None else None, 'epoch': epoch}
        if name is None:
            name = '{}_{}_epoch.pth'.format(os.path.basename(directory), 'last'
                )
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
        num_trainable_params = sum(p.numel() for p in self.parameters() if
            p.requires_grad)
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


class PEXP(nn.Module):

    def __init__(self, n_input, n_out):
        super(PEXP, self).__init__()
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
        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input,
            out_channels=n_input // 2, kernel_size=1), nn.Conv2d(
            in_channels=n_input // 2, out_channels=int(3 * n_input / 4),
            kernel_size=1), nn.Conv2d(in_channels=int(3 * n_input / 4),
            out_channels=int(3 * n_input / 4), kernel_size=3, groups=int(3 *
            n_input / 4), padding=1), nn.Conv2d(in_channels=int(3 * n_input /
            4), out_channels=n_input // 2, kernel_size=1), nn.Conv2d(
            in_channels=n_input // 2, out_channels=n_out, kernel_size=1))

    def forward(self, x):
        return self.network(x)


class CovidNet(nn.Module):

    def __init__(self, model='large', n_classes=3):
        super(CovidNet, self).__init__()
        filters = {'pexp1_1': [64, 256], 'pexp1_2': [256, 256], 'pexp1_3':
            [256, 256], 'pexp2_1': [256, 512], 'pexp2_2': [512, 512],
            'pexp2_3': [512, 512], 'pexp2_4': [512, 512], 'pexp3_1': [512, 
            1024], 'pexp3_2': [1024, 1024], 'pexp3_3': [1024, 1024],
            'pexp3_4': [1024, 1024], 'pexp3_5': [1024, 1024], 'pexp3_6': [
            1024, 1024], 'pexp4_1': [1024, 2048], 'pexp4_2': [2048, 2048],
            'pexp4_3': [2048, 2048]}
        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3))
        for key in filters:
            if 'pool' in key:
                self.add_module(key, nn.MaxPool2d(filters[key][0], filters[
                    key][1]))
            else:
                self.add_module(key, PEXP(filters[key][0], filters[key][1]))
        if model == 'large':
            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64,
                out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256,
                out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512,
                out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024,
                out_channels=2048, kernel_size=1))
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
        pepx11 = self.pexp1_1(x)
        pepx12 = self.pexp1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pexp1_3(pepx12 + pepx11 + out_conv1_1x1)
        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx12 + pepx11 +
            pepx13 + out_conv1_1x1), 2)
        pepx21 = self.pexp2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11,
            2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1, 2))
        pepx22 = self.pexp2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pexp2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pexp2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)
        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx22 + pepx21 +
            pepx23 + pepx24 + out_conv2_1x1), 2)
        pepx31 = self.pexp3_1(F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21,
            2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2) + F.
            max_pool2d(out_conv2_1x1, 2))
        pepx32 = self.pexp3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pexp3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pexp3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pexp3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1
            )
        pepx36 = self.pexp3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 +
            out_conv3_1x1)
        out_conv4_1x1 = F.max_pool2d(self.conv4_1x1(pepx31 + pepx32 +
            pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1), 2)
        pepx41 = self.pexp4_1(F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32,
            2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34, 2) + F.
            max_pool2d(pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(
            out_conv3_1x1, 2))
        pepx42 = self.pexp4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pexp4_3(pepx41 + pepx42 + out_conv4_1x1)
        flattened = self.flatten(pepx41 + pepx42 + pepx43 + out_conv4_1x1)
        fc1out = F.relu(self.fc1(flattened))
        fc2out = F.relu(self.fc2(fc1out))
        logits = self.classifier(fc2out)
        return logits

    def forward_small_net(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        pepx11 = self.pexp1_1(x)
        pepx12 = self.pexp1_2(pepx11)
        pepx13 = self.pexp1_3(pepx12 + pepx11)
        pepx21 = self.pexp2_1(F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11,
            2) + F.max_pool2d(pepx12, 2))
        pepx22 = self.pexp2_2(pepx21)
        pepx23 = self.pexp2_3(pepx22 + pepx21)
        pepx24 = self.pexp2_4(pepx23 + pepx21 + pepx22)
        pepx31 = self.pexp3_1(F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21,
            2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2))
        pepx32 = self.pexp3_2(pepx31)
        pepx33 = self.pexp3_3(pepx31 + pepx32)
        pepx34 = self.pexp3_4(pepx31 + pepx32 + pepx33)
        pepx35 = self.pexp3_5(pepx31 + pepx32 + pepx33 + pepx34)
        pepx36 = self.pexp3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35)
        pepx41 = self.pexp4_1(F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32,
            2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34, 2) + F.
            max_pool2d(pepx35, 2) + F.max_pool2d(pepx36, 2))
        pepx42 = self.pexp4_2(pepx41)
        pepx43 = self.pexp4_3(pepx41 + pepx42)
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

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.2
        ):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
    to keep the spatial dims o=i, this formula is applied
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate=0.2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        norm = nn.BatchNorm3d(num_input_features)
        relu = nn.ReLU(inplace=True)
        conv3d = nn.Conv3d(num_input_features, num_output_features,
            kernel_size=1, padding=0, stride=1)
        self.conv = nn.Sequential(norm, relu, conv3d)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        k = self.conv(x)
        y = self.max_pool(k)
        return y, k


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
        self.add_module('conv', nn.Conv3d(input_features, input_features,
            kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('transp_conv_1', nn.ConvTranspose3d(input_features,
            self.tr_conv1_features, kernel_size=2, padding=0,
            output_padding=0, stride=2))
        self.add_module('transp_conv_2', nn.ConvTranspose3d(self.
            tr_conv1_features, self.tr_conv2_features, kernel_size=2,
            padding=0, output_padding=0, stride=2))


class _HyperDenseLayer(nn.Sequential):

    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,
            num_output_channels, kernel_size=3, stride=1, padding=1, bias=
            False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
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
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1],
                drop_rate)
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
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1],
                drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class ConvInit(nn.Module):

    def __init__(self, in_channels):
        super(ConvInit, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features,
            kernel_size=3, padding=1)
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
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=
            3, padding=1)
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
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=
            3, padding=2, dilation=2)
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
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=
            3, padding=4, dilation=4)
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


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=
    nn.Conv2d, BN=False, ws=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=
        padding, bias=bias)
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

    def __init__(self, nin, nout, bias=False, BN=False, ws=False, activ=nn.
        LeakyReLU(0.2)):
        super(ResidualConv, self).__init__()
        convs = [conv(nin, nout, bias=bias, BN=BN, ws=ws, activ=activ),
            conv(nout, nout, bias=bias, BN=BN, ws=ws, activ=None)]
        self.convs = nn.Sequential(*convs)
        res = []
        if nin != nout:
            res.append(conv(nin, nout, kernel_size=1, padding=0, bias=False,
                BN=BN, ws=ws, activ=None))
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
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride
        =stride, padding=padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None
        ):
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
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None
        ):
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
        transp_conv_1 = nn.ConvTranspose3d(in_channels, transp_channels,
            kernel_size=2, stride=2)
        batch_norm_1 = nn.BatchNorm3d(transp_channels)
        relu_1 = nn.ReLU(inplace=True)
        self.transp_1 = nn.Sequential(transp_conv_1, batch_norm_1, relu_1)
        transp_conv_2 = nn.ConvTranspose3d(transp_channels, transp_channels,
            kernel_size=2, stride=2)
        batch_norm_2 = nn.BatchNorm3d(transp_channels)
        relu_2 = nn.ReLU(inplace=True)
        self.transp_2 = nn.Sequential(transp_conv_2, batch_norm_2, relu_2)
        transp_conv_3 = nn.ConvTranspose3d(transp_channels, transp_channels,
            kernel_size=2, stride=2)
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
        self.layer_2 = nn.Sequential(nn.Conv3d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=(3, 3, 3), stride=1,
            padding=1), norm_2, nn.ReLU())
        self.conv_3 = nn.Conv3d(in_channels=in_channels, out_channels=
            in_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        y = self.conv_3(x)
        y = y + x
        return y


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(3, 3, 3), stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class BlueBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32):
        super(BlueBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpBlock1(nn.Module):
    """
    TODO fix transpose conv to double spatial dim
    """

    def __init__(self, in_channels, out_channels):
        super(UpBlock1, self).__init__()
        self.transp_conv = nn.ConvTranspose3d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=(1, 1, 1), stride=2,
            padding=1)

    def forward(self, x):
        return self.transp_conv(x)


class UpBlock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(1, 1, 1), stride=1)
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
        self.blue_1 = BlueBlock(in_channels=in_channels, out_channels=self.
            start_channels)
        self.drop = nn.Dropout3d(0.2)
        self.green_1 = GreenBlock(in_channels=self.start_channels)
        self.down_1 = DownBlock(in_channels=self.start_channels,
            out_channels=self.down_channels_1)
        self.green_2_1 = GreenBlock(in_channels=self.down_channels_1)
        self.green_2_2 = GreenBlock(in_channels=self.down_channels_1)
        self.down_2 = DownBlock(in_channels=self.down_channels_1,
            out_channels=self.down_channels_2)
        self.green_3_1 = GreenBlock(in_channels=self.down_channels_2)
        self.green_3_2 = GreenBlock(in_channels=self.down_channels_2)
        self.down_3 = DownBlock(in_channels=self.down_channels_2,
            out_channels=self.down_channels_3)
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
        self.up_1 = UpBlock2(in_channels=in_channels, out_channels=
            out_up_1_channels)
        self.green_1 = GreenBlock(in_channels=out_up_1_channels)
        self.up_2 = UpBlock2(in_channels=out_up_1_channels, out_channels=
            out_up_2_channels)
        self.green_2 = GreenBlock(in_channels=out_up_2_channels)
        self.up_3 = UpBlock2(in_channels=out_up_2_channels, out_channels=
            out_up_3_channels)
        self.green_3 = GreenBlock(in_channels=out_up_3_channels)
        self.blue = BlueBlock(in_channels=out_up_3_channels, out_channels=
            classes)

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

    def __init__(self, in_channels=256, in_dim=(10, 10, 10), out_dim=(2, 64,
        64, 64)):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = out_dim[0]
        self.encoder_channels = 16
        self.split_dim = int(self.in_channels / 2)
        self.reshape_dim = int(self.out_dim[1] / self.encoder_channels), int(
            self.out_dim[2] / self.encoder_channels), int(self.out_dim[3] /
            self.encoder_channels)
        self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) *
            (in_dim[1] / 2) * (in_dim[2] / 2))
        self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0
            ] * self.reshape_dim[1] * self.reshape_dim[2]
        channels_vup2 = int(self.in_channels / 2)
        channels_vup1 = int(channels_vup2 / 2)
        channels_vup0 = int(channels_vup1 / 2)
        group_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        relu_1 = nn.ReLU()
        conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=self.
            encoder_channels, stride=2, kernel_size=(3, 3, 3), padding=1)
        self.VD = nn.Sequential(group_1, relu_1, conv_1)
        self.linear_1 = nn.Linear(self.linear_in_dim, in_channels)
        self.linear_vu = nn.Linear(channels_vup2, self.linear_vu_dim)
        relu_vu = nn.ReLU()
        VUup_block = UpBlock2(in_channels=self.encoder_channels,
            out_channels=self.in_channels)
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
        logvar = torch.log(x[:, self.split_dim:])
        y = reparametrize(mu, logvar)
        y = self.linear_vu(y)
        y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.
            reshape_dim[1], self.reshape_dim[2])
        y = self.VU(y)
        y = self.Vup2(y)
        y = self.Vblock2(y)
        y = self.Vup1(y)
        y = self.Vblock1(y)
        y = self.Vup0(y)
        y = self.Vblock0(y)
        dec = self.Vend(y)
        return dec, mu, logvar


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features,
            num_output_features, kernel_size=2, stride=2))


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch,
            out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=
            True))

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
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2))
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
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features,
            kernel_size=5, padding=2)
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
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2,
            kernel_size=2, stride=2)
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_black0017_MedicalZooPytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_001(self):
        self._check(BlueBlock(*[], **{'in_channels': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_002(self):
        self._check(DoubleConv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Down(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(DownBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_005(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(InConv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(LUConv(*[], **{'nchan': 4, 'elu': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_008(self):
        self._check(OutConv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(OutputTransition(*[], **{'in_channels': 4, 'classes': 4, 'elu': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_010(self):
        self._check(PEXP(*[], **{'n_input': 4, 'n_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(ResidualConv(*[], **{'nin': 4, 'nout': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(SkipLastTargetChannelWrapper(*[], **{'loss': MSELoss()}), [torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Up(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    def test_014(self):
        self._check(UpBlock1(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_015(self):
        self._check(UpBlock2(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    @_fails_compile()
    def test_016(self):
        self._check(_MaskingLossWrapper(*[], **{'loss': MSELoss(), 'ignore_index': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

