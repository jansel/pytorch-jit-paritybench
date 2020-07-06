import sys
_module = sys.modules[__name__]
del sys
data = _module
experiment_builder = _module
few_shot_learning_system = _module
inner_loop_optimizers = _module
meta_neural_network_architectures = _module
generate_configs = _module
generate_scripts = _module
train_maml_system = _module
utils = _module
dataset_tools = _module
parser_utils = _module
storage = _module

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


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch


from torchvision import transforms


import time


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import logging


from collections import OrderedDict


import numbers


from copy import copy


from torch import cuda


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, total_num_inner_loop_steps, use_learnable_learning_rates, init_learning_rate=0.001):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        None
        assert init_learning_rate > 0.0, 'learning_rate should be positive.'
        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates

    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace('.', '-')] = nn.Parameter(data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate, requires_grad=self.use_learnable_learning_rates)

    def reset(self):
        pass

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - self.names_learning_rates_dict[key.replace('.', '-')][num_step] * names_grads_wrt_params_dict[key]
        return updated_names_weights_dict


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace('layer_dict.', '')
        name = name.replace('layer_dict.', '')
        name = name.replace('block_dict.', '')
        name = name.replace('module-', '')
        top_level = name.split('.')[0]
        sub_level = '.'.join(name.split('.')[1:])
        if top_level not in output_dict:
            if sub_level == '':
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item
    return output_dict


class MetaBatchNormLayer(nn.Module):

    def __init__(self, num_features, device, args, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, meta_batch_norm=True, no_learnable_params=False, use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta
        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features), requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features), requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=self.learnable_gamma)
        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=self.learnable_gamma)
        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)
        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight, bias = params['weight'], params['bias']
        else:
            weight, bias = self.weight, self.bias
        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None
        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)
        momentum = self.momentum
        output = F.batch_norm(input, running_mean, running_var, weight, bias, training=True, momentum=momentum, eps=self.eps)
        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean, requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var, requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaConv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                weight, bias = params['weight'], params['bias']
            else:
                weight = params['weight']
                bias = None
        elif self.use_bias:
            weight, bias = self.weight, self.bias
        else:
            weight = self.weight
            bias = None
        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaLayerNormLayer(nn.Module):

    def __init__(self, input_feature_shape, eps=1e-05, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = input_feature_shape,
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying a layer norm function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params['bias']
        else:
            bias = self.bias
        return F.layer_norm(input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MetaConvNormLayerReLU(nn.Module):

    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True, meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):
        x = torch.zeros(self.input_shape)
        out = x
        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, use_bias=self.use_bias)
        out = self.conv(out)
        if self.normalization:
            if self.args.norm_layer == 'batch_norm':
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True, meta_batch_norm=self.meta_layer, no_learnable_params=self.no_bn_learnable_params, device=self.device, use_per_step_bn_statistics=self.use_per_step_bn_statistics, args=self.args)
            elif self.args.norm_layer == 'layer_norm':
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])
            out = self.norm_layer(out, num_step=0)
        out = F.leaky_relu(out)
        None

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']
                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']
            conv_params = params['conv']
        out = x
        out = self.conv(out, params=conv_params)
        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step, params=batch_norm_params, training=training, backup_running_statistics=backup_running_statistics)
        out = F.leaky_relu(out)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaLinearLayer(nn.Module):

    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                weight, bias = params['weights'], params['bias']
            else:
                weight = params['weights']
                bias = None
        else:
            pass
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class VGGReLUNormNetwork(nn.Module):

    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes
        if args.max_pooling:
            None
            self.conv_stride = 1
        else:
            None
            self.conv_stride = 2
        self.meta_classifier = meta_classifier
        self.build_network()
        None
        for name, param in self.named_parameters():
            None

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerReLU(input_shape=out.shape, num_filters=self.cnn_filters, kernel_size=3, stride=self.conv_stride, padding=self.args.conv_padding, use_bias=True, args=self.args, normalization=True, meta_layer=self.meta_classifier, no_bn_learnable_params=False, device=self.device)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])
        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)
        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])), num_filters=self.num_output_classes, use_bias=True)
        out = self.layer_dict['linear'](out)
        None

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()
        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split('.')
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
        out = x
        for i in range(self.num_stages):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training, backup_running_statistics=backup_running_statistics, num_step=num_step)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])
        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            None
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            None
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)
    return rng


class MAMLFewShotClassifier(nn.Module):

    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0
        self.rng = set_torch_seed(seed=args.seed)
        self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.num_classes_per_set, args=args, device=device, meta_classifier=True)
        self.task_learning_rate = args.task_learning_rate
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device, init_learning_rate=self.task_learning_rate, total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter, use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))
        None
        for key, value in self.inner_loop_optimizer.named_parameters():
            None
        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self
        None
        for name, param in self.named_parameters():
            if param.requires_grad:
                None
        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs, eta_min=self.args.min_learning_rate)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self
            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=self.args.number_of_training_steps_per_iter) * (1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - self.current_epoch * decay_rate, min_value_for_non_final_losses)
            loss_weights[i] = curr_value
        curr_value = np.minimum(loss_weights[-1] + self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate, 1.0 - (self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses)
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param
                elif 'norm_layer' not in name:
                    param_dict[name] = param
        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)
        grads = torch.autograd.grad(loss, names_weights_copy.values(), create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
        for key, grad in names_grads_copy.items():
            if grad is None:
                None
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy, names_grads_wrt_params_dict=names_grads_copy, num_step=current_step_idx)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {name.replace('module.', ''): value.unsqueeze(0).repeat([num_devices] + [(1) for i in range(len(value.shape))]) for name, value in names_weights_copy.items()}
        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()
        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)
        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        [b, ncs, spc] = y_support_set.shape
        self.num_classes_per_set = ncs
        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_support_set, y_support_set, x_target_set, y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            names_weights_copy = {name.replace('module.', ''): value.unsqueeze(0).repeat([num_devices] + [(1) for i in range(len(value.shape))]) for name, value in names_weights_copy.items()}
            n, s, c, h, w = x_target_set_task.shape
            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)
            for num_step in range(num_steps):
                support_loss, support_preds = self.net_forward(x=x_support_set_task, y=y_support_set_task, weights=names_weights_copy, backup_running_statistics=True if num_step == 0 else False, training=True, num_step=num_step)
                names_weights_copy = self.apply_inner_loop_update(loss=support_loss, names_weights_copy=names_weights_copy, use_second_order=use_second_order, current_step_idx=num_step)
                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task, y=y_target_set_task, weights=names_weights_copy, backup_running_statistics=False, training=True, num_step=num_step)
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == self.args.number_of_training_steps_per_iter - 1:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task, y=y_target_set_task, weights=names_weights_copy, backup_running_statistics=False, training=True, num_step=num_step)
                    task_losses.append(target_loss)
            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)
            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)
            if not training_phase:
                self.classifier.restore_backup_stats()
        losses = self.get_across_task_loss_metrics(total_losses=total_losses, total_accuracies=total_accuracies)
        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()
        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights, training=training, backup_running_statistics=backup_running_statistics, num_step=num_step)
        loss = F.cross_entropy(input=preds, target=y)
        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=self.args.second_order and epoch > self.args.first_order_to_second_order_epoch, use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization, num_steps=self.args.number_of_training_steps_per_iter, training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False, use_multi_step_loss_optimization=True, num_steps=self.args.number_of_evaluation_steps_per_iter, training_phase=False)
        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch
        if not self.training:
            self.train()
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        x_support_set = torch.Tensor(x_support_set).float()
        x_target_set = torch.Tensor(x_target_set).float()
        y_support_set = torch.Tensor(y_support_set).long()
        y_target_set = torch.Tensor(y_target_set).long()
        data_batch = x_support_set, x_target_set, y_support_set, y_target_set
        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()
        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        if self.training:
            self.eval()
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch
        x_support_set = torch.Tensor(x_support_set).float()
        x_target_set = torch.Tensor(x_target_set).float()
        y_support_set = torch.Tensor(y_support_set).long()
        y_target_set = torch.Tensor(y_target_set).long()
        data_batch = x_support_set, x_target_set, y_support_set, y_target_set
        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)
        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, '{}_{}'.format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, learning_rate=0.001):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0.0, 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1) * learning_rate
        self.learning_rate

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.9):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        for key in names_weights_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * names_grads_wrt_params_dict[key]
        return updated_names_weights_dict


class MetaNormLayerConvReLU(nn.Module):

    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True, meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):
        x = torch.zeros(self.input_shape)
        out = x
        if self.normalization:
            if self.args.norm_layer == 'batch_norm':
                self.norm_layer = MetaBatchNormLayer(self.input_shape[1], track_running_stats=True, meta_batch_norm=self.meta_layer, no_learnable_params=self.no_bn_learnable_params, device=self.device, use_per_step_bn_statistics=self.use_per_step_bn_statistics, args=self.args)
            elif self.args.norm_layer == 'layer_norm':
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])
            out = self.norm_layer.forward(out, num_step=0)
        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, use_bias=self.use_bias)
        self.layer_dict['activation_function_pre'] = nn.LeakyReLU()
        out = self.layer_dict['activation_function_pre'].forward(self.conv.forward(out))
        None

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']
            conv_params = params['conv']
        else:
            conv_params = None
        out = x
        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step, params=batch_norm_params, training=training, backup_running_statistics=backup_running_statistics)
        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict['activation_function_pre'].forward(out)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MetaConv2dLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaLayerNormLayer,
     lambda: ([], {'input_feature_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaLinearLayer,
     lambda: ([], {'input_shape': [4, 4], 'num_filters': 4, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_AntreasAntoniou_HowToTrainYourMAMLPytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

