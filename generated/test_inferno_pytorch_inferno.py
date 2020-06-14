import sys
_module = sys.modules[__name__]
del sys
conf = _module
plot_cheap_unet = _module
plot_train_side_loss_unet = _module
plot_unet_tutorial = _module
regularized_mnist = _module
trainer = _module
inferno = _module
extensions = _module
containers = _module
graph = _module
sequential = _module
criteria = _module
core = _module
elementwise_measures = _module
regularized = _module
set_similarity_measures = _module
initializers = _module
base = _module
presets = _module
layers = _module
activations = _module
convolutional = _module
convolutional_blocks = _module
device = _module
identity = _module
normalization = _module
reshape = _module
sampling = _module
metrics = _module
arand = _module
categorical = _module
cremi_score = _module
voi = _module
models = _module
res_unet = _module
unet = _module
optimizers = _module
adam = _module
annealed_adam = _module
ranger = _module
io = _module
box = _module
binary_blobs = _module
camvid = _module
cifar = _module
cityscapes = _module
concatenate = _module
data_utils = _module
zip = _module
transform = _module
generic = _module
image = _module
volume = _module
volumetric = _module
lazy_volume_loader = _module
volumetric_utils = _module
trainers = _module
basic = _module
callbacks = _module
console = _module
essentials = _module
gradients = _module
logging = _module
tensorboard = _module
scheduling = _module
tqdm = _module
tqdmstub = _module
utils = _module
exceptions = _module
io_utils = _module
math_utils = _module
model_utils = _module
partial_cls = _module
python_utils = _module
test_utils = _module
torch_utils = _module
train_utils = _module
version = _module
setup = _module
tests = _module
test_extensions = _module
test_graph = _module
test_core = _module
test_elementwise_measures = _module
test_set_similarity_measures = _module
building_blocks = _module
test_activations = _module
test_convolutional = _module
test_device = _module
test_reshape = _module
test_models = _module
test_res_unet = _module
test_unet = _module
test_inferno = _module
test_io = _module
test_box = _module
test_camvid = _module
test_cityscapes = _module
test_concatenate = _module
test_zip = _module
test_volumetric = _module
test_lazy_volume_loader = _module
test_volume_loader = _module
test_training = _module
test_basic = _module
test_callbacks = _module
test_base = _module
test_essentials = _module
test_logging = _module
test_tensorboard = _module
test_scheduling = _module
test_model_utils = _module
test_partial_cls = _module
test_train_utils = _module

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


import torch


from torch import nn


import numpy


import torch.nn as nn


from collections import OrderedDict


import copy


from torch import nn as nn


from functools import reduce


import warnings


import torch.nn.init as init


import numpy as np


from functools import partial


import torch.nn.functional as F


import functools


from inspect import signature


from numpy import inf


from torch.utils.data import DataLoader


from torch.nn.parallel.data_parallel import data_parallel


from torch.nn import Sequential


from torch.nn import MaxPool2d


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Linear


from torch.nn import Softmax


from torch.utils.data.dataset import TensorDataset


from torch.utils.data.dataloader import DataLoader


class CheapConv(nn.Module):

    def __init__(self, in_channels, out_channels, activated):
        super(CheapConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activated:
            self.convs = torch.nn.Sequential(ConvActivation(in_channels=
                in_channels, out_channels=in_channels, depthwise=True,
                kernel_size=(3, 3), activation='ReLU', dim=2), ConvReLU2D(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(1, 1)))
        else:
            self.convs = torch.nn.Sequential(ConvActivation(in_channels=
                in_channels, out_channels=in_channels, depthwise=True,
                kernel_size=(3, 3), activation='ReLU', dim=2), Conv2D(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(1, 1)))

    def forward(self, x):
        assert x.shape[1
            ] == self.in_channels, 'input has wrong number of channels'
        x = self.convs(x)
        assert x.shape[1
            ] == self.out_channels, 'output has wrong number of channels'
        return x


class CheapConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activated):
        super(CheapConvBlock, self).__init__()
        self.activated = activated
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.start = ConvReLU2D(in_channels=in_channels, out_channels=
                out_channels, kernel_size=(1, 1))
        else:
            self.start = None
        self.conv_a = CheapConv(in_channels=out_channels, out_channels=
            out_channels, activated=True)
        self.conv_b = CheapConv(in_channels=out_channels, out_channels=
            out_channels, activated=False)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x_input = x
        if self.start is not None:
            x_input = self.start(x_input)
        x = self.conv_a(x_input)
        x = self.conv_b(x)
        x = x + x_input
        if self.activated:
            x = self.activation(x)
        return x


def require_dict_kwargs(kwargs, msg=None):
    """ Ensure arguments passed kwargs are either None or a dict.
        If arguments are neither a dict nor None a RuntimeError
        is thrown
    Args:
        kwargs (object): possible dict or None
        msg (None, optional): Error msg

    Returns:
        dict: kwargs dict

    Raises:
        RuntimeError: if the passed value is neither a dict nor None
            this error is raised
    """
    if kwargs is None:
        return dict()
    elif isinstance(kwargs, dict):
        return kwargs
    elif msg is None:
        raise RuntimeError(
            'value passed as keyword argument dict is neither None nor a dict')
    else:
        raise RuntimeError('%s' % str(msg))


class MySideLoss(nn.Module):
    """Wrap a criterion. Collect regularization losses from model and combine with wrapped criterion.
    """

    def __init__(self):
        super(MySideLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=True)
        w = 1.0
        l = None

    def forward(self, predictions, target):
        w = 1.0
        l = None
        for p in predictions:
            ll = self.criterion(p, target) * w
            if l is None:
                l = ll
            else:
                l += ll
            w *= 2
        return l


class RegularizedLinear(nn.Linear):

    def __init__(self, *args, ar_weight=0.001, l1_weight=0.001, **kwargs):
        super(RegularizedLinear, self).__init__(*args, **kwargs)
        self.ar_weight = ar_weight
        self.l1_weight = l1_weight
        self._losses = {}

    def forward(self, input):
        output = super(RegularizedLinear, self).forward(input)
        self._losses['activity_regularization'] = (output * output).sum(
            ) * self.ar_weight
        self._losses['l1_weight_regularization'] = torch.abs(self.weight).sum(
            ) * self.l1_weight
        return output


class Sequential1(nn.Sequential):
    """Like torch.nn.Sequential, but with a few extra methods."""

    def __len__(self):
        return len(self._modules.values())


class Criteria(nn.Module):
    """Aggregate multiple criteria to one."""

    def __init__(self, *criteria):
        super(Criteria, self).__init__()
        if len(criteria) == 1 and isinstance(criteria[0], (list, tuple)):
            criteria = list(criteria[0])
        else:
            criteria = list(criteria)
        assert all([isinstance(criterion, nn.Module) for criterion in criteria]
            ), 'Criterion must be a torch module.'
        self.criteria = criteria

    def forward(self, prediction, target):
        assert isinstance(prediction, (list, tuple)
            ), '`prediction` must be a list or a tuple, got {} instead.'.format(
            type(prediction).__name__)
        assert isinstance(target, (list, tuple)
            ), '`prediction` must be a list or a tuple, got {} instead.'.format(
            type(target).__name__)
        assert len(prediction) == len(target
            ), 'Number of predictions must equal the number of targets. Got {} predictions but {} targets.'.format(
            len(prediction), len(target))
        losses = [criterion(prediction, target) for _prediction, _target,
            criterion in zip(prediction, target, self.criteria)]
        loss = reduce(lambda x, y: x + y, losses)
        return loss


class ShapeError(ValueError):
    pass


class NotTorchModuleError(TypeError):
    pass


def assert_(condition, message='', exception_type=AssertionError):
    """Like assert, but with arbitrary exception types."""
    if not condition:
        raise exception_type(message)


class As2DCriterion(nn.Module):
    """
    Makes a given criterion applicable on (N, C, H, W) prediction and (N, H, W) target tensors,
    if they're applicable to (N, C) prediction and (N,) target tensors .
    """

    def __init__(self, criterion):
        super(As2DCriterion, self).__init__()
        assert_(isinstance(criterion, nn.Module),
            'Criterion must be a module, got a {} instead.'.format(type(
            criterion).__name__), NotTorchModuleError)
        self.criterion = criterion

    def forward(self, prediction, target):
        assert_(prediction.dim() == 4,
            '`prediction` is expected to be a 4D tensor of shape (N, C, H, W), got a {}D tensor instead.'
            .format(prediction.dim()), ShapeError)
        assert_(target.dim() == 3,
            '`target` is expected to be a 3D tensor of shape (N, H, W), got a {}D tensor instead.'
            .format(target.dim()), ShapeError)
        target = target.contiguous().view(-1)
        num_channels = prediction.size(1)
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1,
            num_channels)
        loss = self.criterion(prediction, target)
        return loss


class WeightedMSELoss(nn.Module):
    NEGATIVE_CLASS_WEIGHT = 1.0

    def __init__(self, positive_class_weight=1.0, positive_class_value=1.0,
        size_average=True):
        super(WeightedMSELoss, self).__init__()
        assert_(positive_class_weight >= 0,
            "Positive class weight can't be less than zero, got {}.".format
            (positive_class_weight), ValueError)
        self.mse = nn.MSELoss(size_average=size_average)
        self.positive_class_weight = positive_class_weight
        self.positive_class_value = positive_class_value

    def forward(self, input, target):
        positive_class_mask = target.data.eq(self.positive_class_value
            ).type_as(target.data)
        weight_differential = positive_class_mask.mul_(self.
            positive_class_weight - self.NEGATIVE_CLASS_WEIGHT)
        weights = weight_differential.add_(self.NEGATIVE_CLASS_WEIGHT)
        sqrt_weights = weights.sqrt_()
        return self.mse(input * sqrt_weights, target * sqrt_weights)


def build_criterion(criterion, *args, **kwargs):
    """Build a criterion

    :param criterion: criterion class, name of criterion class, or instance of criterion
    :param args: args for constructor
    :param kwargs: kwargs for constructor
    :return: instance of criterion
    """
    if isinstance(criterion, str):
        for module in [nn, core, set_similarity_measures]:
            criterion_class = getattr(module, criterion, None)
            if criterion_class is not None:
                break
        assert criterion_class is not None, 'Criterion {} not found.'.format(
            criterion)
    elif callable(criterion) and isinstance(criterion, type):
        criterion_class = criterion
    elif isinstance(criterion, torch.nn.Module):
        return criterion
    else:
        raise NotImplementedError
    return criterion_class(*args, **kwargs)


def collect_losses(module):
    """Collect `_losses` dictionaries from module and children

    :param module: a Module to be searched for losses
    :return: dictionary of loss names to values
    """
    losses = {}

    def _collect(m):
        if hasattr(m, '_losses'):
            for k, v in m._losses.items():
                if k in losses:
                    losses[k] = losses[k] + v
                else:
                    losses[k] = v
    module.apply(_collect)
    return losses


class RegularizedLoss(nn.Module):
    """Wrap a criterion. Collect regularization losses from model and combine with wrapped criterion.
    """

    def __init__(self, criterion, *args, **kwargs):
        super(RegularizedLoss, self).__init__()
        self.criterion = build_criterion(criterion, *args, **kwargs)

    def forward(self, *args, trainer=None, model=None, **kwargs):
        main_loss = self.criterion(*args, **kwargs)
        if trainer is None:
            warnings.warn(
                'No trainer parameter provided. Not logging regularization losses.'
                )
        elif model is None:
            model = trainer.model
        if model is None:
            warnings.warn(
                'No model or trainer parameter provided. Not calculating regularization losses.'
                )
            regularization_losses = {}
            total_regularization_loss = None
            total_loss = main_loss
        else:
            regularization_losses = collect_losses(model)
            total_regularization_loss = sum(regularization_losses.values())
            total_loss = main_loss + total_regularization_loss
        if trainer is not None:
            if self.training:
                prefix = 'training'
            else:
                prefix = 'validation'
            updates = {'{}_main_loss'.format(prefix): main_loss}
            if total_regularization_loss is not None:
                updates['{}_total_regularization_loss'.format(prefix)
                    ] = total_regularization_loss
            for k, v in regularization_losses.items():
                updates['{}_{}'.format(prefix, k)] = v
            trainer.update_state_from_dictionary(updates)
        return total_loss


def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert_(input_.dim() >= 2,
        'Tensor or variable must be atleast 2D. Got one of dim {}.'.format(
        input_.dim()), ShapeError)
    num_channels = input_.size(1)
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    permuted = input_.permute(*permute_axes).contiguous()
    flattened = permuted.view(num_channels, -1)
    return flattened


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """

    def __init__(self, weight=None, channelwise=True, eps=1e-06):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        """
        input:      torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        assert input.size() == target.size()
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2.0 * (numerator / denominator.clamp(min=self.eps))
        else:
            input = flatten_samples(input)
            target = flatten_samples(target)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self
                .eps))
            if self.weight is not None:
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                assert self.weight.size() == channelwise_loss.size()
                channelwise_loss = self.weight * channelwise_loss
            loss = channelwise_loss.sum()
        return loss


class GeneralizedDiceLoss(nn.Module):
    """
    Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237

    This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
    one-hot targets for every class.
    """

    def __init__(self, weight=None, channelwise=False, eps=1e-06):
        super(GeneralizedDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        """
        input: torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs:
            - if not channelwise: (batch_size, nb_classes, ...)
            - if channelwise:     (batch_size, nb_channels, nb_classes, ...)
        """
        assert input.size() == target.size()
        if not self.channelwise:
            input = flatten_samples(input)
            target = flatten_samples(target)
            sum_targets = target.sum(-1)
            class_weigths = 1.0 / (sum_targets * sum_targets).clamp(min=
                self.eps)
            numer = ((input * target).sum(-1) * class_weigths).sum()
            denom = ((input + target).sum(-1) * class_weigths).sum()
            loss = 1.0 - 2.0 * numer / denom.clamp(min=self.eps)
        else:

            def flatten_and_preserve_channels(tensor):
                tensor_dim = tensor.dim()
                assert tensor_dim >= 3
                num_channels = tensor.size(1)
                num_classes = tensor.size(2)
                permute_axes = list(range(tensor_dim))
                permute_axes[0], permute_axes[1], permute_axes[2
                    ] = permute_axes[1], permute_axes[2], permute_axes[0]
                permuted = tensor.permute(*permute_axes).contiguous()
                flattened = permuted.view(num_channels, num_classes, -1)
                return flattened
            input = flatten_and_preserve_channels(input)
            target = flatten_and_preserve_channels(target)
            sum_targets = target.sum(-1)
            class_weigths = 1.0 / (sum_targets * sum_targets).clamp(min=
                self.eps)
            numer = ((input * target).sum(-1) * class_weigths).sum(-1)
            denom = ((input + target).sum(-1) * class_weigths).sum(-1)
            channelwise_loss = 1.0 - 2.0 * numer / denom.clamp(min=self.eps)
            if self.weight is not None:
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                assert self.weight.size() == channelwise_loss.size(), """`weight` should have shape (nb_channels, ),
                       `target` should have shape (batch_size, nb_channels, nb_classes, ...)"""
                channelwise_loss = self.weight * channelwise_loss
            loss = channelwise_loss.sum()
        return loss


def where(condition, if_true, if_false):
    """
    Torch equivalent of numpy.where.

    Parameters
    ----------
    condition : torch.ByteTensor or torch.cuda.ByteTensor
        Condition to check.
    if_true : torch.Tensor or torch.cuda.Tensor
        Output value if condition is true.
    if_false: torch.Tensor or torch.cuda.Tensor
        Output value if condition is false

    Returns
    -------
    torch.Tensor

    Raises
    ------
    AssertionError
        if if_true and if_false don't have the same datatype.
    """
    assert if_true.type() == if_false.type(
        ), 'Type mismatch: {} and {}'.format(if_true.data.type(), if_false.
        data.type())
    casted_condition = condition.type_as(if_true)
    output = casted_condition * if_true + (1 - casted_condition) * if_false
    return output


class SELU(nn.Module):

    def forward(self, input):
        return self.selu(input)

    @staticmethod
    def selu(x):
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * where(x >= 0, x, alpha * F.elu(x))


class Initializer(object):
    """
    Base class for all initializers.
    """
    VALID_LAYERS = {'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
        'ConvTranspose2d', 'ConvTranspose3d', 'Linear', 'Bilinear', 'Embedding'
        }

    def __call__(self, module):
        module_class_name = module.__class__.__name__
        if module_class_name in self.VALID_LAYERS:
            try:
                if hasattr(module, 'weight'):
                    self.call_on_weight(module.weight.data)
            except NotImplementedError:
                pass
            try:
                if hasattr(module, 'bias'):
                    self.call_on_bias(module.bias.data)
            except NotImplementedError:
                pass
        return module

    def call_on_bias(self, tensor):
        return self.call_on_tensor(tensor)

    def call_on_weight(self, tensor):
        return self.call_on_tensor(tensor)

    def call_on_tensor(self, tensor):
        raise NotImplementedError

    @classmethod
    def initializes_weight(cls):
        return ('call_on_tensor' in cls.__dict__ or 'call_on_weight' in cls
            .__dict__)

    @classmethod
    def initializes_bias(cls):
        return ('call_on_tensor' in cls.__dict__ or 'call_on_bias' in cls.
            __dict__)


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding by default followed by an activation."""

    def __init__(self, in_channels, out_channels, kernel_size, dim,
        activation, stride=1, dilation=1, groups=None, depthwise=False,
        bias=True, deconv=False, initialization=None, valid_conv=False):
        super(ConvActivation, self).__init__()
        assert_(dim in [1, 2, 3], '`dim` must be one of [1, 2, 3], got {}.'
            .format(dim), ShapeError)
        self.dim = dim
        if depthwise:
            out_channels = in_channels if out_channels in [None, 'auto'
                ] else out_channel
            assert_(in_channels == out_channels,
                'For depthwise convolutions, number of input channels (given: {}) must equal the number of output channels (given {}).'
                .format(in_channels, out_channels), ValueError)
            assert_(groups is None or groups == in_channels,
                'For depthwise convolutions, groups (given: {}) must equal the number of channels (given: {}).'
                .format(groups, in_channels))
            groups = in_channels
        else:
            groups = 1 if groups is None else groups
        self.depthwise = depthwise
        if valid_conv:
            self.conv = getattr(nn, 'Conv{}d'.format(self.dim))(in_channels
                =in_channels, out_channels=out_channels, kernel_size=
                kernel_size, stride=stride, dilation=dilation, groups=
                groups, bias=bias)
        elif not deconv:
            padding = self.get_padding(kernel_size, dilation)
            self.conv = getattr(nn, 'Conv{}d'.format(self.dim))(in_channels
                =in_channels, out_channels=out_channels, kernel_size=
                kernel_size, padding=padding, stride=stride, dilation=
                dilation, groups=groups, bias=bias)
        else:
            self.conv = getattr(nn, 'ConvTranspose{}d'.format(self.dim))(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation,
                groups=groups, bias=bias)
        if initialization is None:
            pass
        elif isinstance(initialization, Initializer):
            self.conv.apply(initialization)
        else:
            raise NotImplementedError
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif activation is None:
            self.activation = None
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        if self.activation is not None:
            activated = self.activation(conved)
        else:
            activated = conved
        return activated

    def _pair_or_triplet(self, object_):
        if isinstance(object_, (list, tuple)):
            assert len(object_) == self.dim
            return object_
        else:
            object_ = [object_] * self.dim
            return object_

    def _get_padding(self, _kernel_size, _dilation):
        assert isinstance(_kernel_size, int)
        assert isinstance(_dilation, int)
        assert _kernel_size % 2 == 1
        return (_kernel_size - 1) // 2 * _dilation

    def get_padding(self, kernel_size, dilation):
        kernel_size = self._pair_or_triplet(kernel_size)
        dilation = self._pair_or_triplet(dilation)
        padding = [self._get_padding(_kernel_size, _dilation) for 
            _kernel_size, _dilation in zip(kernel_size, dilation)]
        return tuple(padding)


class GlobalConv2D(nn.Module):
    """From https://arxiv.org/pdf/1703.02719.pdf
    Main idea: we can have a bigger kernel size computationally acceptable
    if we separate 2D-conv in 2 1D-convs """

    def __init__(self, in_channels, out_channels, kernel_size,
        local_conv_type, activation=None, use_BN=False, **kwargs):
        super(GlobalConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.kwargs = kwargs
        self.conv1a = local_conv_type(in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=(kernel_size[0], 1),
            **kwargs)
        self.conv1b = local_conv_type(in_channels=self.out_channels,
            out_channels=self.out_channels, kernel_size=(1, kernel_size[1]),
            **kwargs)
        self.conv2a = local_conv_type(in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=(1, kernel_size[1]),
            **kwargs)
        self.conv2b = local_conv_type(in_channels=self.out_channels,
            out_channels=self.out_channels, kernel_size=(kernel_size[0], 1),
            **kwargs)
        if use_BN:
            self.batchnorm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batchnorm = None
        self.activation = activation

    def forward(self, input_):
        out1 = self.conv1a(input_)
        out1 = self.conv1b(out1)
        out2 = self.conv2a(input_)
        out2 = self.conv2b(out2)
        out = out1.add(1, out2)
        if self.activation is not None:
            out = self.activation(out)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, layers, resample=None):
        super(ResidualBlock, self).__init__()
        assert pyu.is_listlike(layers)
        self.layers = nn.Sequential(*layers)
        self.resample = resample

    def forward(self, input):
        preaddition = self.layers(input)
        if self.resample is not None:
            skip = self.resample(input)
        else:
            skip = input
        output = preaddition + skip
        return output


def is_listlike(x):
    return isinstance(x, (list, tuple))


def from_iterable(x):
    return x[0] if is_listlike(x) and len(x) == 1 else x


class DeviceError(ValueError):
    pass


class DeviceTransfer(nn.Module):
    """Layer to transfer variables to a specified device."""

    def __init__(self, target_device, device_ordinal=None, asynchronous=False):
        """
        Parameters
        ----------
        target_device : {'cpu', 'cuda'}
            Device to transfer to.
        device_ordinal : int
            Device ordinal if target_device == 'cuda'.
        asynchronous : bool
            Whether to use asynchronous transfers.
        """
        super(DeviceTransfer, self).__init__()
        assert_(target_device in ['cpu', 'cuda'],
            "Target device must either be 'cpu' or 'cuda'.", DeviceError)
        if target_device == 'cpu':
            assert_(device_ordinal is None,
                "'device_ordinal' must be None if target_device is 'cpu'.",
                DeviceError)
        self.target_device = target_device
        self.device_ordinal = device_ordinal

    def forward(self, *inputs):
        if self.target_device == 'cuda':
            transferred = tuple(input_ for input_ in inputs)
        elif self.target_device == 'cpu':
            transferred = tuple(input_.cpu() for input_ in inputs)
        else:
            raise NotImplementedError
        return from_iterable(transferred)


def to_iterable(x):
    return [x] if not is_listlike(x) else x


class OnDevice(nn.Module):
    """
    Moves a module to a device. The advantage of using this over `torch.nn.Module.cuda` is
    that the inputs are transferred to the same device as the module, enabling easy model
    parallelism.
    """

    def __init__(self, module, target_device, device_ordinal=None,
        asynchronous=False):
        """
        Parameters
        ----------
        module : torch.nn.Module
            Module to transfer to device.
        target_device : {'cuda', 'cpu'}
            The device to move `module` to. Must be either 'cuda' or 'cpu'.
        device_ordinal : int
            Ordinal of the GPU device if `target_device = 'cuda'`.
        asynchronous : bool
            Whether to use asynchronous transfers.
        """
        super(OnDevice, self).__init__()
        assert_(target_device in ['cpu', 'cuda'],
            "Target device must either be 'cpu' or 'cuda'.", DeviceError)
        if target_device == 'cpu':
            assert_(device_ordinal is None,
                "'device_ordinal' must be None if target_device is 'cpu'.",
                DeviceError)
        self.target_device = target_device
        self.device_ordinal = device_ordinal
        self.asynchronous = asynchronous
        self.device_transfer = DeviceTransfer(self.target_device,
            device_ordinal=self.device_ordinal, asynchronous=self.asynchronous)
        self.module = self.transfer_module(module)

    def transfer_module(self, module):
        if self.target_device == 'cuda':
            return module
        elif self.target_device == 'cpu':
            return module.cpu()
        else:
            raise NotImplementedError

    def forward(self, *inputs):
        transferred = to_iterable(self.device_transfer(*inputs))
        output = self.module(*transferred)
        return output


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BatchNormND(nn.Module):

    def __init__(self, dim, num_features, eps=1e-05, momentum=0.1, affine=
        True, track_running_stats=True):
        super(BatchNormND, self).__init__()
        assert dim in [1, 2, 3]
        self.bn = getattr(nn, 'BatchNorm{}d'.format(dim))(num_features=
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, x):
        return self.bn(x)


class View(nn.Module):

    def __init__(self, as_shape):
        super(View, self).__init__()
        self.as_shape = self.validate_as_shape(as_shape)

    def validate_as_shape(self, as_shape):
        assert all([(isinstance(_s, int) or _s == 'x') for _s in as_shape])
        all_int_indices = [_n for _n, _s in enumerate(as_shape) if
            isinstance(_s, int)]
        if all_int_indices:
            first_int_at_index = all_int_indices[0]
            assert all([isinstance(_s, int) for _s in as_shape[
                first_int_at_index:]])
        return as_shape

    def forward(self, input):
        input_shape = list(input.size())
        reshaped_shape = [(_s if isinstance(_s, int) else input_shape[_n]) for
            _n, _s in enumerate(self.as_shape)]
        output = input.view(*reshaped_shape)
        return output


class As3D(nn.Module):

    def __init__(self, channel_as_z=False, num_channels_or_num_z_slices=1):
        super(As3D, self).__init__()
        self.channel_as_z = channel_as_z
        self.num_channels_or_num_z_slices = num_channels_or_num_z_slices

    def forward(self, input):
        if input.dim() == 5:
            return input
        elif input.dim() == 4:
            b, c, _0, _1 = list(input.size())
            assert_(c % self.num_channels_or_num_z_slices == 0,
                'Number of channels of the 4D image tensor (= {}) must be divisible by the set number of channels or number of z slices of the 5D volume tensor (= {}).'
                .format(c, self.num_channels_or_num_z_slices), ShapeError)
            c //= self.num_channels_or_num_z_slices
            if self.channel_as_z:
                return input.view(b, self.num_channels_or_num_z_slices, c,
                    _0, _1)
            else:
                return input.view(b, c, self.num_channels_or_num_z_slices,
                    _0, _1)
        elif input.dim() == 2:
            b, c = list(input.size())
            return input.view(b, c, 1, 1, 1)
        else:
            raise NotImplementedError


class As2D(nn.Module):

    def __init__(self, z_as_channel=True):
        super(As2D, self).__init__()
        self.z_as_channel = z_as_channel

    def forward(self, input):
        if input.dim() == 5:
            b, c, _0, _1, _2 = list(input.size())
            if not self.z_as_channel:
                assert _0 == 1
            return input.view(b, c * _0, _1, _2)
        elif input.dim() == 4:
            return input
        elif input.dim() == 2:
            b, c = list(input.size())
            return input.view(b, c, 1, 1)


class Concatenate(nn.Module):
    """Concatenate input tensors along a specified dimension."""

    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class ResizeAndConcatenate(nn.Module):
    """
    Resize input tensors spatially (to a specified target size) before concatenating
    them along the a given dim (channel, i.e. 1 by default). The down-sampling mode can
    be specified ('average' or 'max'), but the up-sampling is always 'nearest'.
    """
    POOL_MODE_MAPPING = {'avg': 'avg', 'average': 'avg', 'mean': 'avg',
        'max': 'max'}

    def __init__(self, target_size, pool_mode='average', dim=1):
        super(ResizeAndConcatenate, self).__init__()
        self.target_size = target_size
        assert_(pool_mode in self.POOL_MODE_MAPPING.keys(),
            '`pool_mode` must be one of {}, got {} instead.'.format(self.
            POOL_MODE_MAPPING.keys(), pool_mode), ValueError)
        self.pool_mode = self.POOL_MODE_MAPPING.get(pool_mode)
        self.dim = dim

    def forward(self, *inputs):
        dim = inputs[0].dim()
        assert_(dim in [4, 5],
            'Input tensors must either be 4 or 5 dimensional, but inputs[0] is {}D.'
            .format(dim), ShapeError)
        spatial_dim = {(4): 2, (5): 3}[dim]
        resize_function = getattr(F, 'adaptive_{}_pool{}d'.format(self.
            pool_mode, spatial_dim))
        target_size = pyu.as_tuple_of_len(self.target_size, spatial_dim)
        resized_inputs = []
        for input_num, input in enumerate(inputs):
            assert_(input.dim() == dim,
                'Expected inputs[{}] to be a {}D tensor, got a {}D tensor instead.'
                .format(input_num, dim, input.dim()), ShapeError)
            resized_inputs.append(resize_function(input, target_size))
        if len(resized_inputs) > 1:
            concatenated = torch.cat(tuple(resized_inputs), self.dim)
        else:
            concatenated = resized_inputs[0]
        return concatenated


class Sum(nn.Module):
    """Sum all inputs."""

    def forward(self, *inputs):
        return torch.stack(inputs, dim=0).sum(0)


class SplitChannels(nn.Module):
    """Split input at a given index along the channel axis."""

    def __init__(self, channel_index):
        super(SplitChannels, self).__init__()
        self.channel_index = channel_index

    def forward(self, input):
        if isinstance(self.channel_index, int):
            split_location = self.channel_index
        elif self.channel_index == 'half':
            split_location = input.size(1) // 2
        else:
            raise NotImplementedError
        assert split_location < input.size(1)
        split_0 = input[:, 0:split_location, (...)]
        split_1 = input[:, split_location:, (...)]
        return split_0, split_1


class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class RemoveSingletonDimension(nn.Module):

    def __init__(self, dim=1):
        super(RemoveSingletonDimension, self).__init__()
        self.dim = 1

    def forward(self, x):
        size = list(x.size())
        if size[self.dim] != 1:
            raise RuntimeError(
                'RemoveSingletonDimension expects a single channel at dim %d, shape=%s'
                 % (self.dim, str(size)))
        slicing = []
        for s in size:
            slicing.append(slice(0, s))
        slicing[self.dim] = 0
        return x[slicing]


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest',
        align_corners=None):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        super(Upsample, self).__init__()
        if hasattr(nn.functional, 'interpolate'):
            self.have_interpolate = True
        else:
            self.have_interpolate = False
            self.sampler = nn.Upsample(size=size, scale_factor=scale_factor,
                mode=mode, align_corners=align_corners)

    def forward(self, input):
        if self.have_interpolate:
            return nn.functional.interpolate(input, self.size, self.
                scale_factor, self.mode, self.align_corners)
        else:
            return self.sampler(input)


class AnisotropicUpsample(nn.Module):

    def __init__(self, scale_factor):
        super(AnisotropicUpsample, self).__init__()
        self.upsampler = Upsample(scale_factor=scale_factor)

    def forward(self, input):
        N, C, D, H, W = input.size()
        folded = input.view(N, C * D, H, W)
        upsampled = self.upsampler(folded)
        unfolded = upsampled.view(N, C, D, self.upsampler.scale_factor * H,
            self.upsampler.scale_factor * W)
        return unfolded


class AnisotropicPool(nn.MaxPool3d):

    def __init__(self, downscale_factor):
        ds = downscale_factor
        super(AnisotropicPool, self).__init__(kernel_size=(1, ds + 1, ds + 
            1), stride=(1, ds, ds), padding=(0, 1, 1))


class AnisotropicUpsample2D(nn.Module):

    def __init__(self, scale_factor):
        super(AnisotropicUpsample2D, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=scale_factor)

    def forward(self, input):
        N, C, D, W = input.size()
        folded = input.view(N, C * D, W)
        upsampled = self.upsampler(folded)
        unfolded = upsampled.view(N, C, D, self.upsampler.scale_factor * W)
        return unfolded


class AnisotropicPool2D(nn.MaxPool2d):

    def __init__(self, downscale_factor):
        ds = downscale_factor
        super(AnisotropicPool2D, self).__init__(kernel_size=(1, ds + 1),
            stride=(1, ds), padding=(0, 1))


class _ResBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, dim, size=2,
        force_skip_op=False, activated=True):
        super(_ResBlockBase, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.size = int(size)
        self.activated = bool(activated)
        self.force_skip_op = bool(force_skip_op)
        self.dim = int(dim)
        if self.in_channels != self.out_channels or self.force_skip_op:
            self.activated_skip_op = self.activated_skip_op_factory(in_channels
                =self.in_channels, out_channels=self.out_channels)
        conv_ops = []
        activation_ops = []
        for i in range(self.size):
            if i == 0:
                op = self.nonactivated_conv_op_factory(in_channels=self.
                    out_channels, out_channels=self.out_channels, index=i)
            else:
                op = self.nonactivated_conv_op_factory(in_channels=self.
                    out_channels, out_channels=self.out_channels, index=i)
            conv_ops.append(op)
            if i < self.size or self.activated:
                activation_ops.append(self.activation_op_factory(index=i))
        self.conv_ops = nn.ModuleList(conv_ops)
        self.activation_ops = nn.ModuleList(activation_ops)

    def activated_skip_op_factory(self, in_channels, out_channels):
        raise NotImplementedError(
            'activated_skip_op_factory need to be implemented by deriving class'
            )

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        raise NotImplementedError(
            'conv_op_factory need to be implemented by deriving class')

    def activation_op_factory(self, index):
        return nn.ReLU()

    def forward(self, input):
        if input.size(1) != self.in_channels:
            raise RuntimeError(
                'wrong number of channels: expected %d, got %d' % (self.
                in_channels, input.size(1)))
        if input.dim() != self.dim + 2:
            raise RuntimeError('wrong number of dim: expected %d, got %d' %
                (self.dim + 2, input.dim()))
        if self.in_channels != self.out_channels or self.force_skip_op:
            skip_res = self.activated_skip_op(input)
        else:
            skip_res = input
        assert skip_res.size(1) == self.out_channels
        res = skip_res
        for i in range(self.size):
            res = self.conv_ops[i](res)
            assert res.size(1) == self.out_channels
            if i + 1 < self.size:
                res = self.activation_ops[i](res)
        non_activated = skip_res + res
        if self.activated:
            return self.activation_ops[-1](non_activated)
        else:
            return non_activated


def max_allowed_ds_steps(shape, factor):
    """How often can a shape be down-sampled by a given factor
        such that non of the divisions will give non-integers.

    Args:
        shape (listlike): tensor shape
        factor (integer): downsample factor

    Returns:
        int: maximum allowed downsample operations
    """

    def max_allowed_ds_steps_impl(size, factor):
        current_size = float(size)
        allowed_steps = 0
        while True:
            new_size = current_size / float(factor)
            if new_size >= 1 and new_size.is_integer():
                current_size = new_size
                allowed_steps += 1
            else:
                break
        return allowed_steps
    min_steps = float('inf')
    for s in shape:
        min_steps = int(min(min_steps, max_allowed_ds_steps_impl(s, factor)))
    return min_steps


class UNetBase(nn.Module):
    """ Base class for implementing UNets.
        The depth and dimension of the UNet is flexible.
        The deriving classes must implement
        `conv_op_factory` and can implement
        `upsample_op_factory` and
        `downsample_op_factory`.

    Attributes:
        in_channels (int): Number of input channels.
        dim (int): Spatial dimension of data (must be 2 or 3).
        out_channels (int): Number of output channels. Set to None by default,
            which sets the number of out channels to the number of input channels
            to preserve symmetry of feature channels (default: None).
        depth (int): How many down-sampling / up-sampling steps
            shall be performed (default: 3).
        gain (int): Multiplicative increase of channels while going down in the UNet.
            The same factor is used to decrease the number of channels while
            going up in the UNet (default: 2).
        residual (bool): If residual is true, the output of the down-streams
            are added to the up-stream results.
            Otherwise the results are concatenated (default: False).
    """

    def __init__(self, in_channels, dim, out_channels=None, depth=3, gain=2,
        residual=False, upsample_mode=None, p_dropout=None):
        super(UNetBase, self).__init__()
        if dim not in [2, 3]:
            raise RuntimeError('UNetBase is only implemented for 2D and 3D')
        self.in_channels = int(in_channels)
        self.dim = int(dim)
        self.out_channels = self.in_channels if out_channels is None else int(
            out_channels)
        self.depth = int(depth)
        self.gain = int(gain)
        self.residual = bool(residual)
        self.p_dropout = p_dropout
        self._store_conv_down = []
        self._store_conv_bottom = False
        self._store_conv_up = []
        self.n_channels_per_output = []
        self._pre_conv_down_ops = None
        self._post_conv_down_ops = None
        self._conv_down_ops = None
        self._pre_conv_up_ops = None
        self._post_conv_up_ops = None
        self._conv_up_ops = None
        self._upsample_ops = None
        self._downsample_ops = None
        self._pre_conv_bottom_ops = None
        self._post_conv_bottom_ops = None
        self._conv_bottom_op = None
        self._upsample_kwargs = self._make_upsample_kwargs(upsample_mode=
            upsample_mode)
        if self.p_dropout is not None:
            self.use_dropout = True
            if self.dim == 2:
                self._channel_dropout_op = self.torch.nn.Dropout2d(p=float(
                    self.p_dropout), inplace=False)
            else:
                self._channel_dropout_op = self.torch.nn.Dropout3d(p=float(
                    self.p_dropout), inplace=False)
        else:
            self.use_dropout = False
        self._init__downstream()
        self._downsample_ops = nn.ModuleList([self.downsample_op_factory(i) for
            i in range(depth)])
        self._upsample_ops = nn.ModuleList([self.upsample_op_factory(depth -
            i - 1) for i in range(depth)])
        self._init__bottom()
        self._init__upstream()
        assert len(self.n_channels_per_output) == self._store_conv_down.count(
            True) + self._store_conv_up.count(True) + int(self.
            _store_conv_bottom)

    def _get_num_channels(self, depth):
        assert depth > 0
        return self.in_channels * self.gain ** depth

    def _init__downstream(self):
        conv_down_ops = []
        self._store_conv_down = []
        current_in_channels = self.in_channels
        for i in range(self.depth):
            out_channels = self._get_num_channels(i + 1)
            op, return_op_res = self.conv_op_factory(in_channels=
                current_in_channels, out_channels=out_channels, part='down',
                index=i)
            conv_down_ops.append(op)
            if return_op_res:
                self.n_channels_per_output.append(out_channels)
                self._store_conv_down.append(True)
            else:
                self._store_conv_down.append(False)
            current_in_channels = out_channels
        self._conv_down_ops = nn.ModuleList(conv_down_ops)
        return current_in_channels

    def _init__bottom(self):
        current_in_channels = self._get_num_channels(self.depth)
        factory_res = self.conv_op_factory(in_channels=current_in_channels,
            out_channels=current_in_channels, part='bottom', index=0)
        if isinstance(factory_res, tuple):
            self._conv_bottom_op, self._store_conv_bottom = factory_res
            if self._store_conv_bottom:
                self.n_channels_per_output.append(current_in_channels)
        else:
            self._conv_bottom_op = factory_res
            self._store_conv_bottom = False

    def _init__upstream(self):
        conv_up_ops = []
        current_in_channels = self._get_num_channels(self.depth)
        for i in range(self.depth):
            out_channels = (self.out_channels if i + 1 == self.depth else
                self._get_num_channels(self.depth - i - 1))
            fac = 1 if self.residual else 2
            op, return_op_res = self.conv_op_factory(in_channels=fac *
                current_in_channels, out_channels=out_channels, part='up',
                index=self.depth - i - 1)
            conv_up_ops.append(op)
            if return_op_res:
                self.n_channels_per_output.append(out_channels)
                self._store_conv_up.append(True)
            else:
                self._store_conv_up.append(False)
            current_in_channels = out_channels
        self._conv_up_ops = nn.ModuleList(conv_up_ops)
        if not self._store_conv_up[-1]:
            self._store_conv_up[-1] = True
            self.n_channels_per_output.append(out_channels)

    def _make_upsample_kwargs(self, upsample_mode):
        """To avoid some waring from pytorch, and some missing implementations
        for the arguments need to be handle carefully in this helper functions

        Args:
            upsample_mode (str): users choice for upsampling  interpolation style.
        """
        if upsample_mode is None:
            if self.dim == 2:
                upsample_mode = 'bilinear'
            elif self.dim == 3:
                upsample_mode = 'trilinear'
        upsample_kwargs = dict(scale_factor=2, mode=upsample_mode)
        if upsample_mode in ('bilinear', 'trilinear'):
            upsample_kwargs['align_corners'] = False
        return upsample_kwargs

    def _forward_sanity_check(self, input):
        if isinstance(input, tuple):
            raise RuntimeError('tuples of tensors are not supported')
        shape = input.shape
        if shape[1] != self.in_channels:
            raise RuntimeError(
                'wrong number of channels: expected %d, got %d' % (self.
                in_channels, input.size(1)))
        if input.dim() != self.dim + 2:
            raise RuntimeError('wrong number of dim: expected %d, got %d' %
                (self.dim + 2, input.dim()))
        self._check_scaling(input)

    def _check_scaling(self, input):
        shape = input.shape
        mx = max_allowed_ds_steps(shape=shape[2:2 + self.dim], factor=2)
        if mx < self.depth:
            raise RuntimeError('cannot downsample %d times, with shape %s' %
                (self.depth, str(input.size())))

    def forward(self, input):
        self._forward_sanity_check(input=input)
        side_out = []
        down_res = []
        out = input
        for d in range(self.depth):
            out = self._conv_down_ops[d](out)
            down_res.append(out)
            if self._store_conv_down[d]:
                side_out.append(out)
            out = self._downsample_ops[d](out)
        out = self._conv_bottom_op(out)
        if self._store_conv_bottom:
            side_out.append(out)
        down_res = list(reversed(down_res))
        for d in range(self.depth):
            out = self._upsample_ops[d](out)
            a = down_res[d]
            if self.residual:
                out = a + out
            else:
                out = torch.cat([a, out], 1)
            out = self._conv_up_ops[d](out)
            if self._store_conv_up[d]:
                side_out.append(out)
        if len(side_out) == 1:
            return side_out[0]
        else:
            return tuple(side_out)

    def downsample_op_factory(self, index):
        C = nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
        return C(kernel_size=2, stride=2)

    def upsample_op_factory(self, index):
        return InfernoUpsample(**self._upsample_kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):
        raise NotImplementedError(
            'conv_op_factory need to be implemented by deriving class')

    def _dropout(self, x):
        if self.use_dropout:
            return self._channel_dropout_op(x)
        else:
            return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_inferno_pytorch_inferno(_paritybench_base):
    pass
    def test_000(self):
        self._check(As2D(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(As3D(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GeneralizedDiceLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(RegularizedLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(SELU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Sequential1(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(SorensenDiceLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(Squeeze(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(View(*[], **{'as_shape': [4, 4]}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(WeightedMSELoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

