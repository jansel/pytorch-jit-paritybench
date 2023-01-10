import sys
_module = sys.modules[__name__]
del sys
conf = _module
ezflow = _module
config = _module
retrieve = _module
data = _module
dataloader = _module
dataloader_creator = _module
device_dataloader = _module
dataset = _module
autoflow = _module
base_dataset = _module
driving = _module
flying_chairs = _module
flying_things3d = _module
hd1k = _module
kitti = _module
kubric = _module
monkaa = _module
mpi_sintel = _module
decoder = _module
build = _module
context = _module
conv_decoder = _module
iterative = _module
recurrent_lookup = _module
noniterative = _module
operators = _module
soft_regression = _module
pyramid = _module
separable_conv = _module
encoder = _module
build = _module
conv_encoder = _module
ganet = _module
pspnet = _module
pyramid = _module
residual = _module
engine = _module
eval = _module
profiler = _module
pruning = _module
registry = _module
trainer = _module
functional = _module
criterion = _module
multiscale = _module
sequence = _module
data_augmentation = _module
augmentor = _module
operations = _module
scheduler = _module
model_zoo = _module
models = _module
build = _module
dicl = _module
flownet_c = _module
flownet_s = _module
predictor = _module
pwcnet = _module
raft = _module
vcn = _module
modules = _module
base_module = _module
blocks = _module
dap = _module
recurrent = _module
units = _module
similarity = _module
correlation = _module
layer = _module
pairwise = _module
sampler = _module
learnable_cost = _module
utils = _module
io = _module
metrics = _module
other_utils = _module
resampling = _module
viz = _module
warp = _module
generate_dir_structure = _module
setup = _module
tests = _module
test_decoder = _module
test_encoder = _module
test_engine = _module
test_functional = _module
test_models = _module
test_modules = _module
test_similarity = _module
test_utils = _module
mock_data = _module
mock_model = _module
train = _module

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


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.distributed import DistributedSampler


import random


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


import math


import time


from torch.nn import DataParallel


from torch.profiler import profile


from torch.profiler import record_function


from torch.profiler import ProfilerActivity


from torch.profiler import schedule


from torch.profiler import tensorboard_trace_handler


from torch.nn.utils import prune


from torch.nn import CrossEntropyLoss


from torch.nn import L1Loss


from torch.nn import MSELoss


from torch.optim import SGD


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import RMSprop


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import OneCycleLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from copy import deepcopy


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


import numbers


import scipy.ndimage as ndimage


import torchvision


from torch.nn import functional as F


from torchvision.transforms import ColorJitter


import torch.optim as optim


from torch.nn.init import constant_


from torch.nn.init import kaiming_normal_


from torchvision import io


from typing import Tuple


from typing import Union


import re


from scipy import interpolate


from torch.utils.data import DataLoader


from torchvision import transforms as T


from torch.utils.data import Dataset


class Registry:
    """
    Class to register objects and then retrieve them by name.

    Parameters
    ----------
    name : str
        Name of the registry
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Method to register an object in the registry

        Parameters
        ----------
        obj : object, optional
            Object to register, defaults to None (which will return the decorator)
        name : str, optional
            Name of the object to register, defaults to None (which will use the name of the object)
        """
        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        """
        Method to retrieve an object from the registry

        Parameters
        ----------
        name : str
            Name of the object to retrieve

        Returns
        -------
        object
            Object registered under the given name
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def get_list(self):
        """
        Method to retrieve all objects from the registry

        Returns
        -------
        list
            List of all objects registered in the registry
        """
        return list(self._obj_map.keys())

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())


DECODER_REGISTRY = Registry('DECODER')


def _called_with_cfg(*args, **kwargs):
    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop('cfg', None), (_CfgNode, DictConfig)):
        return True
    return False


def _get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != 'cfg':
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f'{from_config_func.__self__}.from_config'
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD] for param in signature.parameters.values())
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.

    """
    if init_func is not None:
        assert inspect.isfunction(init_func) and from_config is None and init_func.__name__ == '__init__', 'Incorrect use of @configurable. Check API documentation for examples.'

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError("Class with @configurable must have a 'from_config' classmethod.") from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")
            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
        return wrapped
    else:
        if from_config is None:
            return configurable
        assert inspect.isfunction(from_config), 'from_config argument of configurable must be a function!'

        def wrapper(orig_func):

            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            wrapped.from_config = from_config
            return wrapped
        return wrapper


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=None):
    """
    2D convolution layer with optional normalization followed by
    an inplace LeakyReLU activation of 0.1 negative slope.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, default: 3
        Size of the convolutional kernel
    stride : int, default: 1
        Stride of the convolutional kernel
    padding : int, default: 1
        Padding of the convolutional kernel
    dilation : int, default: 1
        Dilation of the convolutional kernel
    norm : str, default: None
        Type of normalization to use. Can be None, 'batch', 'layer', 'group'
    """
    bias = False
    if norm == 'group':
        norm_fn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
    elif norm == 'batch':
        norm_fn = nn.BatchNorm2d(out_channels)
    elif norm == 'instance':
        norm_fn = nn.InstanceNorm2d(out_channels)
    else:
        norm_fn = nn.Identity()
        bias = True
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias), norm_fn, nn.LeakyReLU(0.1, inplace=True))


class ContextNetwork(nn.Module):
    """
    PWCNet Context Network decoder

    Parameters
    ----------
    in_channels: int, default: 565
        Number of input channels
    config : List[int], default : [128, 128, 96, 64, 32]
        List containing all output channels of the decoder.
    """

    @configurable
    def __init__(self, in_channels=565, config=[128, 128, 96, 64, 32]):
        super(ContextNetwork, self).__init__()
        self.context_net = nn.ModuleList([conv(in_channels, config[0], kernel_size=3, stride=1, padding=1, dilation=1)])
        self.context_net.append(conv(config[0], config[0], kernel_size=3, stride=1, padding=2, dilation=2))
        self.context_net.append(conv(config[0], config[1], kernel_size=3, stride=1, padding=4, dilation=4))
        self.context_net.append(conv(config[1], config[2], kernel_size=3, stride=1, padding=8, dilation=8))
        self.context_net.append(conv(config[2], config[3], kernel_size=3, stride=1, padding=16, dilation=16))
        self.context_net.append(conv(config[3], config[4], kernel_size=3, stride=1, padding=1, dilation=1))
        self.context_net.append(nn.Conv2d(config[4], 2, kernel_size=3, stride=1, padding=1, bias=True))
        self.context_net = nn.Sequential(*self.context_net)

    @classmethod
    def from_config(self, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'config': cfg.CONFIG}

    def forward(self, x):
        return self.context_net(x)


class ConvDecoder(nn.Module):
    """
    Applies a 2D Convolutional decoder to the input feature map.
    Used in **PWCNet** (https://arxiv.org/abs/1709.02371)

    Parameters
    ----------
    config : List[int], default : [128, 128, 96, 64, 32]
        List containing all output channels of the decoder
    concat_channels : int, optional
        Additional input channels to be concatenated for convolution layers
    to_flow : bool, default : True
        If True, convoloves decoder output to optical flow of shape N x 2 x H x W
    block : object, default : None
        the conv block to be used to build the decoder layers.
    """

    @configurable
    def __init__(self, config=[128, 128, 96, 64, 32], concat_channels=None, to_flow=True, block=None):
        super().__init__()
        self.concat_channels = concat_channels
        if block is None:
            block = conv
        self.decoder = nn.ModuleList()
        config_cumsum = torch.cumsum(torch.tensor(config), dim=0)
        if concat_channels is not None:
            self.decoder.append(block(concat_channels, config[0], kernel_size=3, stride=1))
        for i in range(len(config) - 1):
            if concat_channels is not None:
                in_channels = config_cumsum[i] + concat_channels
            else:
                in_channels = config[i]
            self.decoder.append(block(in_channels, config[i + 1], kernel_size=3, stride=1))
        self.to_flow = nn.Identity()
        if to_flow:
            if concat_channels is not None:
                in_channels = config_cumsum[-1] + concat_channels
            else:
                in_channels = config[-1]
            self.to_flow = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)

    @classmethod
    def from_config(self, cfg):
        return {'config': cfg.CONFIG}

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W representing the flow

        torch.Tensor
            Tensor of shape N x output_channel x H x W
        """
        for i in range(len(self.decoder)):
            y = self.decoder[i](x)
            if self.concat_channels is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y
        return self.to_flow(x), x


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    """
    2D transpose convolution layer followed by an activation function

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int, optional
        Size of the convolutional kernel
    stride : int, optional
        Stride of the convolutional kernel
    padding : int, optional
        Padding of the convolutional kernel
    dilation : int, optional
        Dilation of the convolutional kernel
    """
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)


class FlowNetConvDecoder(nn.Module):
    """
    Applies a 2D Convolutional decoder to regress the optical flow
    from the intermediate outputs convolutions of the encoder.
    Used in **FlowNetSimple** (https://arxiv.org/abs/1504.06852)

    Parameters
    ----------
    in_channels : int, default: 1024
        Number of input channels of the decoder. This value should be equal to the final output channels of the encoder
    config : List[int], default : [512, 256, 128, 64]
        List containing all output channels of the decoder
    """

    @configurable
    def __init__(self, in_channels=1024, config=[512, 256, 128, 64]):
        super().__init__()
        if isinstance(config, tuple):
            config = list(config)
        out_channels = [in_channels] + config
        in_channels = []
        prev_out_channels = 0
        for i in range(len(out_channels)):
            if i > 0:
                inp = out_channels[i] + prev_out_channels + 2
                prev_out_channels = out_channels[i]
            else:
                inp = out_channels[i]
                prev_out_channels = out_channels[i + 1]
            in_channels.append(inp)
        self.predict_flow = nn.ModuleList()
        self.upsample_flow = nn.ModuleList()
        self.deconv = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.predict_flow.append(nn.Conv2d(in_channels[i], 2, kernel_size=3, stride=1, padding=1, bias=False))
            self.upsample_flow.append(nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False))
            self.deconv.append(deconv(in_channels[i], out_channels[i + 1]))
        self.to_flow = nn.Conv2d(in_channels[-1], 2, kernel_size=3, stride=1, padding=1, bias=False)

    @classmethod
    def from_config(self, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'config': cfg.CONFIG}

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : List[torch.Tensor]
            List of all the outputs from each convolution layer of the encoder

        Returns
        -------
        List[torch.Tensor],
            List of all the flow predictions from each decoder layer
        """
        flow_preds = []
        conv_out = x[-1]
        flow = self.predict_flow[0](conv_out)
        flow_up = self.upsample_flow[0](flow)
        deconv_out = self.deconv[0](conv_out)
        flow_preds.append(flow)
        layer_index = 1
        start = len(x) - 2
        end = 1
        for conv_out in x[start:end:-1]:
            assert conv_out.shape[2] == deconv_out.shape[2] == flow_up.shape[2]
            assert conv_out.shape[3] == deconv_out.shape[3] == flow_up.shape[3]
            concat_out = torch.cat((conv_out, deconv_out, flow_up), dim=1)
            flow = self.predict_flow[layer_index](concat_out)
            flow_up = self.upsample_flow[layer_index](flow)
            deconv_out = self.deconv[layer_index](concat_out)
            flow_preds.append(flow)
            layer_index += 1
        concat_out = torch.cat((x[1], deconv_out, flow_up), dim=1)
        flow = self.to_flow(concat_out)
        flow_preds.append(flow)
        return flow_preds


class FlowHead(nn.Module):
    """
    Applies two 2D convolutions over an input feature map
    to generate a flow tensor of shape N x 2 x H x W.

    Parameters
    ----------
    input_dim : int, default: 128
        Number of input dimensions.
    hidden_dim : int, default: 256
        Number of hidden dimensions.
    """

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape N x input_dim x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W
        """
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    """
    Applies two Convolution GRU cells to the input signal.
    Each GRU cell uses separate convolution layers.

    Parameters
    ----------
    hidden_dim : int, default: 128
        Number of hidden dimensions.
    input_dim : int, default: 192 + 128
        Number of hidden dimensions.
    """

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        """
        Performs forward pass.

        Parameters
        ----------
        h : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representating the hidden state

        x : torch.Tensor
            A tensor of shape N x input_dim + hidden_dim x H x W representating the input


        Returns
        -------
        torch.Tensor
            a tensor of shape N x hidden_dim x H x W
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):
    """
    Encodes motion features from the correlation levels of the pyramid
    and the input flow estimate using convolution layers.


    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid

    """

    def __init__(self, corr_radius, corr_levels):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """
        Parameters
        ----------
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W

        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 82 x H x W
        """
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class MotionEncoder(nn.Module):
    """
    Encodes motion features from the correlation levels of the pyramid
    and the input flow estimate using convolution layers.


    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid

    """

    def __init__(self, corr_radius, corr_levels):
        super(MotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """
        Parameters
        ----------
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W

        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 128 x H x W
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


MODULE_REGISTRY = Registry('MODULE')


class ConvGRU(nn.Module):
    """
    Convolutinal GRU layer

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension of the GRU
    input_dim : int, optional
        Input dimension of the GRU
    kernel_size : int, optional
        Kernel size of the convolutional layers
    """

    @configurable
    def __init__(self, hidden_dim=128, input_dim=192 + 128, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=1)

    @classmethod
    def from_config(cls, cfg):
        return {'hidden_dim': cfg.HIDDEN_DIM, 'input_dim': cfg.INPUT_DIM, 'kernel_size': cfg.KERNEL_SIZE}

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallRecurrentLookupUpdateBlock(nn.Module):
    """
    Applies an iterative lookup update on all levels of the correlation
    pyramid to estimate flow with a sequence of GRU cells.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid
    hidden_dim  : int, default: 96
        Number of hidden dimensions.
    input_dim   : int, default: 64
        Number of input dimensions.
    """

    @configurable
    def __init__(self, corr_radius, corr_levels, hidden_dim=96, input_dim=64):
        super(SmallRecurrentLookupUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_radius, corr_levels)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    @classmethod
    def from_config(cls, cfg):
        return {'corr_radius': cfg.CORR_RADIUS, 'corr_levels': cfg.CORR_LEVELS, 'hidden_dim': cfg.HIDDEN_DIM, 'input_dim': cfg.INPUT_DIM}

    def forward(self, net, inp, corr, flow):
        """
        Performs forward pass.

        Parameters
        ----------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W
        inp : torch.Tensor
            A tensor of shape N x input_dim x H x W
        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W


        Returns
        -------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representing the output of the GRU cell
        mask : NoneType
        delta_flow : torch.Tensor
            A tensor of shape N x 2 x H x W representing the delta flow
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


class RecurrentLookupUpdateBlock(nn.Module):
    """
    Applies an iterative lookup update on all levels of the correlation
    pyramid to estimate flow with a sequence of GRU cells.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid
    hidden_dim  : int, default: 128
        Number of hidden dimensions.
    input_dim   : int, default: 128
        Number of input dimensions.
    """

    @configurable
    def __init__(self, corr_radius, corr_levels, hidden_dim=128, input_dim=128):
        super(RecurrentLookupUpdateBlock, self).__init__()
        self.encoder = MotionEncoder(corr_radius, corr_levels)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    @classmethod
    def from_config(cls, cfg):
        return {'corr_radius': cfg.CORR_RADIUS, 'corr_levels': cfg.CORR_LEVELS, 'hidden_dim': cfg.HIDDEN_DIM, 'input_dim': cfg.INPUT_DIM}

    def forward(self, net, inp, corr, flow):
        """
        Performs forward pass.

        Parameters
        ----------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W
        inp : torch.Tensor
            A tensor of shape N x input_dim x H x W
        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W


        Returns
        -------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representing the output of the SepConvGRU cell.
        mask : torch.Tensor
            A tensor of shape N x 576 x H x W
        delta_flow : torch.Tensor
            A tensor of shape N x 2 x H x W representing the delta flow
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class FlowEntropy(nn.Module):
    """
    Computes entropy from matching cost

    """

    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape B x U x V x H x W representing the cost

        Returns
        -------
        torch.Tensor
            A tensor of shape B x 1 x H x W
        """
        x = torch.squeeze(x, 1)
        B, U, V, H, W = x.shape
        x = x.view(B, -1, H, W)
        x = F.softmax(x, dim=1).view(B, U, V, H, W)
        global_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
        global_entropy /= np.log(x.shape[1] * x.shape[2])
        return global_entropy


class SoftArg2DFlowRegression(nn.Module):
    """
    Applies 2D soft argmin/argmax operation to regress flow.
    Used in **DICL** (https://arxiv.org/abs/2010.14851)

    Parameters
    ----------
    max_u : int, default : 3
        Maximum displacement in the horizontal direction
    max_v : int, default : 3
        Maximum displacement in the vertical direction
    operation : str, default : argmax
        The argmax/argmin operation for flow regression
    """

    @configurable
    def __init__(self, max_u=3, max_v=3, operation='argmax'):
        super(SoftArg2DFlowRegression, self).__init__()
        assert operation.lower() == 'argmax' or operation.lower() == 'argmin', 'Invalid operation. Supported operations: argmax and argmin'
        self.max_u = max_u
        self.max_v = max_v
        self.operation = operation.lower()

    @classmethod
    def from_config(cls, cfg):
        return {'max_u': cfg.MAX_U, 'max_v': cfg.MAX_V, 'operation': cfg.OPERATION}

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W representing the flow
        """
        sizeU = 2 * self.max_u + 1
        sizeV = 2 * self.max_v + 1
        x = x.squeeze(1)
        B, _, _, H, W = x.shape
        disp_u = torch.reshape(torch.arange(-self.max_u, self.max_u + 1, dtype=torch.float32), [1, sizeU, 1, 1, 1])
        disp_u = disp_u.expand(B, -1, sizeV, H, W).contiguous()
        disp_u = disp_u.view(B, sizeU * sizeV, H, W)
        disp_v = torch.reshape(torch.arange(-self.max_v, self.max_v + 1, dtype=torch.float32), [1, 1, sizeV, 1, 1])
        disp_v = disp_v.expand(B, sizeU, -1, H, W).contiguous()
        disp_v = disp_v.view(B, sizeU * sizeV, H, W)
        x = x.view(B, sizeU * sizeV, H, W)
        if self.operation == 'argmin':
            x = F.softmin(x, dim=1)
        else:
            x = F.softmax(x, dim=1)
        flow_u = (x * disp_u).sum(dim=1)
        flow_v = (x * disp_v).sum(dim=1)
        flow = torch.cat((flow_u.unsqueeze(1), flow_v.unsqueeze(1)), dim=1)
        return flow


class Soft4DFlowRegression(nn.Module):
    """
    Applies 4D soft argmax operation to regress flow.

    Parameters
    ----------
    size : List[int]
        List containing values of B, H, W
    max_disp : int, default : 4
        Maximum displacement
    entropy : bool, default : False
        If True, computes local and global entropy from matching cost
    factorization : int, default : 1
        Max displacement factorization value
    """

    @configurable
    def __init__(self, size, max_disp=4, entropy=False, factorization=1):
        super(Soft4DFlowRegression, self).__init__()
        B, H, W = size
        self.entropy = entropy
        self.md = max_disp
        self.factorization = factorization
        self.truncated = True
        self.w_size = 3
        flowrange_y = range(-max_disp, max_disp + 1)
        flowrange_x = range(-int(max_disp // self.factorization), int(max_disp // self.factorization) + 1)
        meshgrid = np.meshgrid(flowrange_x, flowrange_y)
        flow_y = np.tile(np.reshape(meshgrid[0], [1, 2 * max_disp + 1, 2 * int(max_disp // self.factorization) + 1, 1, 1]), (B, 1, 1, H, W))
        flow_x = np.tile(np.reshape(meshgrid[1], [1, 2 * max_disp + 1, 2 * int(max_disp // self.factorization) + 1, 1, 1]), (B, 1, 1, H, W))
        self.register_buffer('flow_x', torch.Tensor(flow_x))
        self.register_buffer('flow_y', torch.Tensor(flow_y))
        self.pool3d = nn.MaxPool3d((self.w_size * 2 + 1, self.w_size * 2 + 1, 1), stride=1, padding=(self.w_size, self.w_size, 0))

    @classmethod
    def from_config(cls, cfg):
        return {'size': cfg.SIZE, 'max_disp': cfg.MAX_DISP, 'entropy': cfg.ENTROPY, 'factorization': cfg.FACTORIZATION}

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input cost feature map of shape B x U x V x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape B x C x H x W representing the flow

        torch.Tensor
            A tensor representing the local and global entropy cost
        """
        B, U, V, H, W = x.shape
        orig_x = x
        if self.truncated:
            x = x.view(B, U * V, H, W)
            idx = x.argmax(1)[:, np.newaxis]
            mask = torch.FloatTensor(B, U * V, H, W).fill_(0)
            mask.scatter_(1, idx, 1)
            mask = mask.view(B, 1, U, V, -1)
            mask = self.pool3d(mask)[:, 0].view(B, U, V, H, W)
            n_inf = x.clone().fill_(-np.inf).view(B, U, V, H, W)
            x = torch.where(mask.byte(), orig_x, n_inf)
        else:
            self.w_size = (np.sqrt(U * V) - 1) / 2
        B, U, V, H, W = x.shape
        x = F.softmax(x.view(B, -1, H, W), 1).view(B, U, V, H, W)
        out_x = torch.sum(torch.sum(x * self.flow_x, 1), 1, keepdim=True)
        out_y = torch.sum(torch.sum(x * self.flow_y, 1), 1, keepdim=True)
        if self.entropy:
            local_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
            if self.w_size == 0:
                local_entropy[:] = 1.0
            else:
                local_entropy /= np.log((self.w_size * 2 + 1) ** 2)
            x = F.softmax(orig_x.view(B, -1, H, W), 1).view(B, U, V, H, W)
            global_entropy = (-x * torch.clamp(x, 1e-09, 1 - 1e-09).log()).sum(1).sum(1)[:, np.newaxis]
            global_entropy /= np.log(x.shape[1] * x.shape[2])
            return torch.cat([out_x, out_y], 1), torch.cat([local_entropy, global_entropy], 1)
        else:
            return torch.cat([out_x, out_y], 1), None


def warp(x, flow):
    """
    Warps an image x according to the optical flow field specified

    Parameters
    ----------
    x : torch.Tensor
        Image to be warped
    flow : torch.Tensor
        Optical flow field

    Returns
    -------
    torch.Tensor
        Warped image
    """
    B, _, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = torch.Tensor(grid) + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones_like(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


class PyramidDecoder(nn.Module):
    """
    Applies a 2D Convolutional decoder to regress the optical flow
    from the intermediate outputs convolutions of the encoder.
    Used in **PWCNet** (https://arxiv.org/abs/1709.02371)

    Parameters
    ----------
    config : List[int], default : [128, 128, 96, 64, 32]
        List containing all output channels of the decoder.
    to_flow : bool, default : True
        If True, regresses the flow of shape N x 2 x H x W.
    max_displacement: int, default: 4
        Maximum displacement for cost volume computation.
    pad_size: int, default: 0
        Pad size for cost volume computation.
    flow_scale_factor: float, default: 20.0
        Scale factor for upscaling flow predictions.
    """

    @configurable
    def __init__(self, config=[128, 128, 96, 64, 32], to_flow=True, max_displacement=4, pad_size=0, flow_scale_factor=20.0):
        super(PyramidDecoder, self).__init__()
        self.config = config
        self.flow_scale_factor = flow_scale_factor
        self.correlation_layer = SpatialCorrelationSampler(kernel_size=1, patch_size=2 * max_displacement + 1, padding=pad_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        search_range = (2 * max_displacement + 1) ** 2
        self.decoder_layers = nn.ModuleList()
        self.up_feature_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        for i in range(len(config)):
            if i == 0:
                concat_channels = search_range
            else:
                concat_channels = search_range + config[i] + max_displacement
            self.decoder_layers.append(ConvDecoder(config=config, to_flow=to_flow, concat_channels=concat_channels))
            if i < len(config) - 1:
                self.deconv_layers.append(deconv(2, 2, kernel_size=4, stride=2, padding=1))
                self.up_feature_layers.append(deconv(concat_channels + sum(config), 2, kernel_size=4, stride=2, padding=1))

    @classmethod
    def from_config(self, cfg):
        return {'config': cfg.CONFIG, 'to_flow': cfg.TO_FLOW, 'max_displacement': cfg.SIMILARITY.MAX_DISPLACEMENT, 'pad_size': cfg.SIMILARITY.PAD_SIZE, 'flow_scale_factor': cfg.FLOW_SCALE_FACTOR}

    def _corr_relu(self, features1, features2):
        corr = self.correlation_layer(features1, features2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        return self.leaky_relu(corr)

    def forward(self, feature_pyramid1, feature_pyramid2):
        """
        Performs forward pass.

        Parameters
        ----------
        feature_pyramid1 : torch.Tensor
            Input feature map of image 1

        feature_pyramid2 : torch.Tensor
            Input feature map of image 2

        Returns
        -------
        List[torch.Tensor]
            A List containing tensors of shape N x 2 x H x W representing the flow

        List[torch.Tensor]
            A List containing tensors of shape N x output_channel x H x W
        """
        up_flow, up_features = None, None
        up_flow_scale = self.flow_scale_factor * 2 ** -len(self.config)
        flow_preds = []
        for i in range(len(self.decoder_layers)):
            if i == 0:
                corr = self._corr_relu(feature_pyramid1[i], feature_pyramid2[i])
                concatenated_features = corr
            else:
                warped_features = warp(feature_pyramid2[i], up_flow * up_flow_scale)
                up_flow_scale *= 2
                corr = self._corr_relu(feature_pyramid1[i], warped_features)
                concatenated_features = torch.cat([corr, feature_pyramid1[i], up_flow, up_features], dim=1)
            flow, features = self.decoder_layers[i](concatenated_features)
            flow_preds.append(flow)
            if i < len(self.decoder_layers) - 1:
                up_flow = self.deconv_layers[i](flow)
                up_features = self.up_feature_layers[i](features)
        return flow_preds, features


class FeatureProjection4D(nn.Module):
    """
    Applies a 3D convolution to the input feature map

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Norm 3D
    groups : int, default : 1
        Number of groupds for 3D convolution, in_channels and out_channels must be divisible by groups
    """

    def __init__(self, in_channels, out_channels, stride, norm=True, groups=1):
        super(FeatureProjection4D, self).__init__()
        self.norm = norm
        self.stride = stride
        bias = not norm
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1, (stride, stride, 1), padding=0, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        B, C, U, V, H, W = x.size()
        x = self.conv1(x.view(B, C, U, V, H * W))
        if self.norm:
            x = self.bn(x)
        _, C, U, V, _ = x.shape
        x = x.view(B, C, U, V, H, W)
        return x


class SeparableConv4D(nn.Module):
    """
    Applies two 3D convolution followed by an
    optional 2D convolution to the input feature map.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    k_size : int, default : 3
        Size of the kernel
    full : bool, default : True
        If True, applies a stride of (1, 1, 1)
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    @configurable
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), norm=True, k_size=3, full=True, groups=1):
        super(SeparableConv4D, self).__init__()
        bias = not norm
        self.is_proj = False
        self.stride = stride[0]
        expand = 1
        if norm:
            if in_channels != out_channels:
                self.is_proj = True
                self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=bias, padding=0, groups=groups), nn.BatchNorm2d(out_channels))
            if full:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels * expand, in_channels, (1, k_size, k_size), stride=(1, self.stride, self.stride), bias=bias, padding=(0, k_size // 2, k_size // 2), groups=groups), nn.BatchNorm3d(in_channels))
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_channels * expand, in_channels, (1, k_size, k_size), stride=1, bias=bias, padding=(0, k_size // 2, k_size // 2), groups=groups), nn.BatchNorm3d(in_channels))
            self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels * expand, (k_size, k_size, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(k_size // 2, k_size // 2, 0), groups=groups), nn.BatchNorm3d(in_channels * expand))
        else:
            if in_channels != out_channels:
                self.is_proj = True
                self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias, padding=0, groups=groups)
            if full:
                self.conv1 = nn.Conv3d(in_channels * expand, in_channels, (1, k_size, k_size), stride=(1, self.stride, self.stride), bias=bias, padding=(0, k_size // 2, k_size // 2), groups=groups)
            else:
                self.conv1 = nn.Conv3d(in_channels * expand, in_channels, (1, k_size, k_size), stride=1, bias=bias, padding=(0, k_size // 2, k_size // 2), groups=groups)
            self.conv2 = nn.Conv3d(in_channels, in_channels * expand, (k_size, k_size, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(k_size // 2, k_size // 2, 0), groups=groups)
        self.relu = nn.ReLU(inplace=True)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS, 'stride': cfg.STRIDE, 'norm': cfg.NORM, 'k_size': cfg.k_size, 'full': cfg.FULL, 'groups': cfg.GROUPS}

    def forward(self, x):
        B, C, U, V, H, W = x.shape
        x = self.conv2(x.view(B, C, U, V, -1))
        B, C, U, V, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(B, C, -1, H, W))
        B, C, _, H, W = x.shape
        if self.is_proj:
            x = self.proj(x.view(B, W, -1, W))
        x = x.view(B, -1, U, V, H, W)
        return x


class SeparableConv4DBlock(nn.Module):
    """
    Applies separate SeperableConv4d convolutions to the input feature map.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    k_size : int, default : 3
        Size of the kernel
    full : bool, default : True
        If True, applies SeparableConv4D otherwise FeatureProjection4D
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), norm=True, full=True, groups=1):
        super(SeparableConv4DBlock, self).__init__()
        if in_channels == out_channels and stride == (1, 1, 1):
            self.downsample = None
        elif full:
            self.downsample = SeparableConv4D(in_channels, out_channels, stride, norm=norm, k_size=1, full=full, groups=groups)
        else:
            self.downsample = FeatureProjection4D(in_channels, out_channels, stride[0], norm=norm, groups=groups)
        self.conv1 = SeparableConv4D(in_channels, out_channels, stride, norm=norm, full=full, groups=groups)
        self.conv2 = SeparableConv4D(out_channels, out_channels, (1, 1, 1), norm=norm, full=full, groups=groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        if self.downsample:
            x = self.downsample(x)
        out = self.relu2(x + self.conv2(out))
        return out


class Butterfly4D(nn.Module):
    """
    Applies a FeatureProjection4D followed by
    five SeperableConv4d convolutions to the input feature map.

    Parameters
    ----------
    f_dim_1 : int
        Number of input channels
    f_dim_2 : int
        Number of output channels
    stride : tuple, default : (1, 1, 1)
        Stride of the convolution
    norm : bool, default : True
        If True, applies Batch Normalization
    full : bool, default : True
        If True, applies SeparableConv4D otherwise FeatureProjection4D
    groups : int, default : 1
        Number of groups for 3D convolution, in_channels and out_channels must both be divisible by groups
    """

    @configurable
    def __init__(self, f_dim_1, f_dim_2, norm=True, full=True, groups=1):
        super(Butterfly4D, self).__init__()
        self.proj = nn.Sequential(FeatureProjection4D(f_dim_1, f_dim_2, 1, norm=norm, groups=groups), nn.ReLU(inplace=True))
        self.conva1 = SeparableConv4DBlock(f_dim_2, f_dim_2, norm=norm, stride=(2, 1, 1), full=full, groups=groups)
        self.conva2 = SeparableConv4DBlock(f_dim_2, f_dim_2, norm=norm, stride=(2, 1, 1), full=full, groups=groups)
        self.convb3 = SeparableConv4DBlock(f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups)
        self.convb2 = SeparableConv4DBlock(f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups)
        self.convb1 = SeparableConv4DBlock(f_dim_2, f_dim_2, norm=norm, stride=(1, 1, 1), full=full, groups=groups)

    @classmethod
    def from_config(cls, cfg):
        return {'f_dim_1': cfg.F_DIM_1, 'f_dim_2': cfg.F_DIM_2, 'norm': cfg.NORM, 'full': cfg.FULL, 'groups': cfg.GROUPS}

    def forward(self, x):
        out = self.proj(x)
        B, C, U, V, H, W = out.shape
        out1 = self.conva1(out)
        _, _, U1, V1, H1, W1 = out1.shape
        out2 = self.conva2(out1)
        _, _, U2, V2, H2, W2 = out2.shape
        out2 = self.convb3(out2)
        t_out_1 = F.interpolate(out2.view(B, C, U2, V2, -1), (U1, V1, H2 * W2), mode='trilinear', align_corners=True).view(B, C, U1, V1, H2, W2)
        t_out_1 = F.interpolate(t_out_1.view(B, C, -1, H2, W2), (U1 * V1, H1, W1), mode='trilinear', align_corners=True).view(B, C, U1, V1, H1, W1)
        out1 = t_out_1 + out1
        out1 = self.convb2(out1)
        t_out = F.interpolate(out1.view(B, C, U1, V1, -1), (U, V, H1 * W1), mode='trilinear', align_corners=True).view(B, C, U, V, H1, W1)
        t_out = F.interpolate(t_out.view(B, C, -1, H1, W1), (U * V, H, W), mode='trilinear', align_corners=True).view(B, C, U, V, H, W)
        out = t_out + out
        out = self.convb1(out)
        return out


ENCODER_REGISTRY = Registry('ENCODER')


class BasicConvEncoder(nn.Module):
    """
    A Basic Convolution Encoder with a fixed size kernel = 3, padding=1 and dilation = 1.
    Every alternate layer has stride = 1 followed by stride = 2.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration for the layers in the encoder
    norm : str
        Type of normalization to use. Can be None, 'batch', 'group', 'instance'
    """

    @configurable
    def __init__(self, in_channels=3, config=[64, 128, 256, 512], norm=None):
        super(BasicConvEncoder, self).__init__()
        if isinstance(config, tuple):
            config = list(config)
        channels = [in_channels] + config
        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 1 if i % 2 == 0 else 2
            kernel_size = 3
            self.encoder.append(conv(channels[i], channels[i + 1], kernel_size=3, stride=stride, padding=(kernel_size - 1) // 2, norm=norm))

    @classmethod
    def from_config(self, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'config': cfg.CONFIG, 'norm': cfg.NORM}

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        List[torch.Tensor],
            List of all the output convolutions from each encoder layer
        """
        outputs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if len(outputs) > 0:
                prev_output = outputs[-1]
                if prev_output.shape[1:] == x.shape[1:]:
                    outputs[-1] = x
                else:
                    outputs.append(x)
            else:
                outputs.append(x)
        return outputs


class FlowNetConvEncoder(BasicConvEncoder):
    """
    Convolutional encoder based on the FlowNet architecture
    Used in **FlowNet: Learning Optical Flow with Convolutional Networks** (https://arxiv.org/abs/1504.06852)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration for the layers in the encoder
    norm : str
        Type of normalization to use. Can be None, 'batch', 'group', 'instance'
    """

    @configurable
    def __init__(self, in_channels=3, config=[64, 128, 256, 512], norm=None):
        super(FlowNetConvEncoder, self).__init__()
        assert len(config) >= 2, 'FlowNetConvEncoder expects at least 2 output channels in config.'
        if isinstance(config, tuple):
            config = list(config)
        channels = [in_channels] + config
        self.encoder = nn.ModuleList()
        self.encoder.append(conv(channels[0], channels[1], kernel_size=7, stride=2, padding=(7 - 1) // 2))
        self.encoder.append(conv(channels[1], channels[2], kernel_size=5, stride=2, padding=(5 - 1) // 2))
        self.encoder.append(conv(channels[2], channels[3], kernel_size=5, stride=2, padding=(5 - 1) // 2))
        channels = channels[3:]
        for i in range(len(channels) - 1):
            stride = 1 if i % 2 == 0 else 2
            kernel_size = 3
            self.encoder.append(conv(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, norm=norm))

    @classmethod
    def from_config(self, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'config': cfg.CONFIG, 'norm': cfg.NORM}


class ConvNormRelu(nn.Module):
    """
    Block for a convolutional layer with normalization and activation

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    deconv : bool, optional
        If True, use a transposed convolution
    norm : str, optional
        Normalization method
    activation : str, optional
        Activation function
    """

    def __init__(self, in_channels, out_channels, deconv=False, norm='batch', activation='relu', **kwargs):
        super(ConvNormRelu, self).__init__()
        bias = False
        if norm is not None:
            if norm.lower() == 'group':
                self.norm = nn.GroupNorm(out_channels)
            elif norm.lower() == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm.lower() == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            bias = True
        if activation is not None:
            if activation.lower() == 'leakyrelu':
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()
        if 'kernel_size' not in kwargs.keys():
            kwargs['kernel_size'] = 3
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=bias, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv2x(nn.Module):
    """
    Double convolutional layer with the option to perform deconvolution and concatenation

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    deconv : bool
        Whether to perform deconvolution instead of convolution
    concat : bool
        Whether to concatenate the input and the output of the first convolution layer
    norm : str
        Type of normalization to use. Can be "batch", "instance", "group", or "none"
    activation : str
        Type of activation to use. Can be "relu" or "leakyrelu"
    """

    def __init__(self, in_channels, out_channels, deconv=False, concat=True, norm='batch', activation='relu'):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.deconv = deconv
        if deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = ConvNormRelu(in_channels, out_channels, deconv, kernel_size=kernel, stride=2, padding=1)
        if self.concat:
            self.conv2 = ConvNormRelu(out_channels * 2, out_channels, deconv=False, norm=norm, activation=activation, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = ConvNormRelu(out_channels, out_channels, deconv=False, norm=norm, activation=activation, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class GANetBackbone(nn.Module):
    """
    Feature extractor backbone used in **GA-Net: Guided Aggregation Net for End-to-end Stereo Matching** (https://arxiv.org/abs/1904.06587)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """

    @configurable
    def __init__(self, in_channels=3, out_channels=32):
        super(GANetBackbone, self).__init__()
        self.conv_start = nn.Sequential(ConvNormRelu(in_channels, 32, kernel_size=3, padding=1), ConvNormRelu(32, 32, kernel_size=3, stride=2, padding=1), ConvNormRelu(32, 32, kernel_size=3, padding=1))
        self.conv1a = ConvNormRelu(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = ConvNormRelu(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = ConvNormRelu(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = ConvNormRelu(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = ConvNormRelu(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = ConvNormRelu(160, 192, kernel_size=3, stride=2, padding=1)
        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)
        self.deconv6b = Conv2x(192, 160, deconv=True)
        self.outconv_6 = ConvNormRelu(160, 32, kernel_size=3, padding=1)
        self.deconv5b = Conv2x(160, 128, deconv=True)
        self.outconv_5 = ConvNormRelu(128, 32, kernel_size=3, padding=1)
        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.outconv_4 = ConvNormRelu(96, 32, kernel_size=3, padding=1)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.outconv_3 = ConvNormRelu(64, 32, kernel_size=3, padding=1)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.outconv_2 = ConvNormRelu(48, out_channels, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS}

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x
        x = self.deconv6a(x, rem5)
        rem5 = x
        x = self.deconv5a(x, rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x
        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)
        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)
        x = self.deconv5b(x, rem4)
        x5 = self.outconv_5(x)
        x = self.deconv4b(x, rem3)
        x4 = self.outconv_4(x)
        x = self.deconv3b(x, rem2)
        x3 = self.outconv_3(x)
        x = self.deconv2b(x, rem1)
        x2 = self.outconv_2(x)
        return [x, x2, x3, x4, x5, x6]


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1, norm=True):
        super(ResidualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.downsample = nn.Identity() if downsample is None else downsample
        if norm:
            norm = 'batch'
        else:
            norm = None
        self.block = nn.Sequential(ConvNormRelu(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, norm=norm), ConvNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm, activation=None))

    def forward(self, x):
        residual = x
        out = self.block(x)
        residual = self.downsample(x)
        out += residual
        out = F.leaky_relu(out, 0.1)
        return out


class PyramidPooling(nn.Module):
    """
    Pyramid pooling module for the **PSPNet** feature extractor

    Parameters
    ----------
    in_channels : int
        Number of input channels
    levels : int
        Number of levels in the pyramid
    norm : bool
        Whether to use batch normalization
    """

    def __init__(self, in_channels, levels=4, norm=True):
        super(PyramidPooling, self).__init__()
        self.levels = levels
        if norm:
            norm = 'batch'
        else:
            norm = None
        self.paths = []
        for _ in range(levels):
            self.paths.append(ConvNormRelu(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, norm=norm))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        H, W = x.shape[2:]
        k_sizes = []
        strides = []
        for pool_size in torch.linspace(1, min(H, W) // 2, self.levels):
            k_sizes.append((int(H / pool_size), int(W / pool_size)))
            strides.append((int(H / pool_size), int(W / pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]
        pp_sum = x
        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
            pp_sum = pp_sum + 1.0 / self.levels * out
        pp_sum = self.relu(pp_sum / 2.0)
        return pp_sum


class PSPNetBackbone(nn.Module):
    """
    PSPNet feature extractor backbone (https://arxiv.org/abs/1612.01105)
    Used in **Volumetric Correspondence Networks for Optical Flow** (https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)

    Parameters
    ----------
    is_proj : bool
        Whether to use projection pooling or not
    groups : int
        Number of groups in the convolutional
    in_channels : int
        Number of input channels
    norm : bool
        Whether to use batch normalization

    """

    @configurable
    def __init__(self, is_proj=True, groups=1, in_channels=3, norm=True):
        super(PSPNetBackbone, self).__init__()
        self.is_proj = is_proj
        self.inplanes = 32
        if norm:
            norm = 'batch'
        else:
            norm = None
        self.convbnrelu1_1 = ConvNormRelu(in_channels, 16, kernel_size=3, padding=1, stride=2, norm=norm)
        self.convbnrelu1_2 = ConvNormRelu(16, 16, kernel_size=3, padding=1, stride=1, norm=norm)
        self.convbnrelu1_3 = ConvNormRelu(16, 32, kernel_size=3, padding=1, stride=1, norm=norm)
        self.res_block3 = self._make_layer(ResidualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(ResidualBlock, 128, 1, stride=2)
        self.pyramid_pooling = PyramidPooling(128, levels=3, norm=norm)
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2), ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm))
        self.iconv5 = ConvNormRelu(192, 128, kernel_size=3, padding=1, stride=1, norm=norm)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2), ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm))
        self.iconv4 = ConvNormRelu(192, 128, kernel_size=3, padding=1, stride=1, norm=norm)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2), ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm))
        self.iconv3 = ConvNormRelu(128, 64, kernel_size=3, padding=1, stride=1, norm=norm)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2), ConvNormRelu(64, 32, kernel_size=3, padding=1, stride=1, norm=norm))
        self.iconv2 = ConvNormRelu(64, 64, kernel_size=3, padding=1, stride=1, norm=norm)
        if self.is_proj:
            self.proj6 = ConvNormRelu(128, 128 // groups, kernel_size=1, padding=0, stride=1)
            self.proj5 = ConvNormRelu(128, 128 // groups, kernel_size=1, padding=0, stride=1)
            self.proj4 = ConvNormRelu(128, 128 // groups, kernel_size=1, padding=0, stride=1)
            self.proj3 = ConvNormRelu(64, 64 // groups, kernel_size=1, padding=0, stride=1)
            self.proj2 = ConvNormRelu(64, 64 // groups, kernel_size=1, padding=0, stride=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg):
        return {'is_proj': cfg.IS_PROJ, 'groups': cfg.GROUPS, 'in_channels': cfg.IN_CHANNELS, 'norm': cfg.NORM}

    def forward(self, x):
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)
        pool1 = F.max_pool2d(conv1, 3, 2, 1)
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)
        conv6x = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=True)
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)
        conv5x = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=True)
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)
        conv4x = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=True)
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)
        conv3x = F.interpolate(conv3, [pool1.size()[2], pool1.size()[3]], mode='bilinear', align_corners=True)
        concat2 = torch.cat((pool1, self.upconv3[1](conv3x)), dim=1)
        conv2 = self.iconv2(concat2)
        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return [proj6, proj5, proj4, proj3, proj2]
        return [conv6, conv5, conv4, conv3, conv2]


class PyramidEncoder(nn.Module):
    """
    Pyramid encoder which returns a hierarchy of features
    Used in **PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume** (https://arxiv.org/abs/1709.02371)

    Parameters
    ----------
    in_channels : int
        Number of input channels
    config : list of int
        Configuration of the pyramid encoder's layers
    """

    @configurable
    def __init__(self, in_channels=3, config=[16, 32, 64, 96, 128, 196]):
        super().__init__()
        if isinstance(config, tuple):
            config = list(config)
        config = [in_channels] + config
        self.encoder = nn.ModuleList()
        for i in range(len(config) - 1):
            self.encoder.append(nn.Sequential(conv(config[i], config[i + 1], kernel_size=3, stride=2), conv(config[i + 1], config[i + 1], kernel_size=3, stride=1), conv(config[i + 1], config[i + 1], kernel_size=3, stride=1)))

    @classmethod
    def from_config(self, cfg):
        return {'config': cfg.CONFIG}

    def forward(self, img):
        """
        Performs forward pass.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor

        Returns
        -------
        List[torch.Tensor],
            List of all the output convolutions from each encoder layer
        """
        feature_pyramid = []
        x = img
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            feature_pyramid.append(x)
        return feature_pyramid


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-style architectures

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Stride of the convolution
    norm : str, optional
        Normalization method. One of "group", "batch", "instance", or None
    activation : str, optional
        Activation function. One of "relu", "leakyrelu", or None
    """

    @configurable
    def __init__(self, in_channels, out_channels, stride=1, norm='group', activation='relu'):
        super(BasicBlock, self).__init__()
        if norm is not None:
            if norm.lower() == 'group':
                n_groups = out_channels // 8
                norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
                norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
                if stride != 1:
                    norm3 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
            elif norm.lower() == 'batch':
                norm1 = nn.BatchNorm2d(out_channels)
                norm2 = nn.BatchNorm2d(out_channels)
                if stride != 1:
                    norm3 = nn.BatchNorm2d(out_channels)
            elif norm.lower() == 'instance':
                norm1 = nn.InstanceNorm2d(out_channels)
                norm2 = nn.InstanceNorm2d(out_channels)
                if stride != 1:
                    norm3 = nn.InstanceNorm2d(out_channels)
            elif norm.lower() == 'none':
                norm1 = nn.Identity()
                norm2 = nn.Identity()
                if stride != 1:
                    norm3 = nn.Identity()
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()
            if stride != 1:
                norm3 = nn.Identity()
        if activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.residual_fn = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride), norm1, self.activation, nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), norm2)
        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), norm3)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS, 'stride': cfg.STRIDE, 'norm': cfg.NORM, 'activation': cfg.ACTIVATION}

    def forward(self, x):
        out = self.residual_fn(x)
        out = self.activation(out + self.shortcut(x))
        return out


class BasicEncoder(nn.Module):
    """
    ResNet-style encoder with basic residual blocks

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    norm : str
        Normalization layer to use. One of "batch", "instance", "group", or None
    p_dropout : float
        Dropout probability
    layer_config : list of int or tuple of int
        Configuration of encoder's layers
    intermediate_features : bool
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(self, in_channels=3, out_channels=128, norm='batch', p_dropout=0.0, layer_config=(64, 96, 128), intermediate_features=False):
        super(BasicEncoder, self).__init__()
        self.intermediate_features = intermediate_features
        norm = norm.lower()
        assert norm in ('group', 'batch', 'instance', 'none')
        start_channels = layer_config[0]
        if norm == 'group':
            norm_fn = nn.GroupNorm(num_groups=8, num_channels=start_channels)
        elif norm == 'batch':
            norm_fn = nn.BatchNorm2d(start_channels)
        elif norm == 'instance':
            norm_fn = nn.InstanceNorm2d(start_channels)
        elif norm == 'none':
            norm_fn = nn.Identity()
        layers = nn.ModuleList([nn.Conv2d(in_channels, start_channels, kernel_size=7, stride=2, padding=3), norm_fn, nn.ReLU(inplace=True)])
        for i in range(len(layer_config)):
            if i == 0:
                stride = 1
            else:
                stride = 2
            layers.append(self._make_layer(start_channels, layer_config[i], stride, norm))
            start_channels = layer_config[i]
        layers.append(nn.Conv2d(layer_config[-1], out_channels, kernel_size=1))
        dropout = nn.Identity()
        if self.training and p_dropout > 0:
            dropout = nn.Dropout2d(p=p_dropout)
        layers.append(dropout)
        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm):
        layer1 = BasicBlock(in_channels, out_channels, stride, norm)
        layer2 = BasicBlock(out_channels, out_channels, stride=1, norm=norm)
        return nn.Sequential(*[layer1, layer2])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS, 'norm': cfg.NORM, 'p_dropout': cfg.P_DROPOUT, 'layer_config': cfg.LAYER_CONFIG, 'intermediate_features': cfg.INTERMEDIATE_FEATURES}

    def forward(self, x):
        if self.intermediate_features is True:
            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)
                if isinstance(self.encoder[i], nn.Sequential):
                    features.append(x)
            features.append(x)
            return features
        else:
            is_list = isinstance(x, tuple) or isinstance(x, list)
            if is_list:
                batch_dim = x[0].shape[0]
                x = torch.cat(x, dim=0)
            out = self.encoder(x)
            if is_list:
                out = torch.split(out, [batch_dim, batch_dim], dim=0)
            return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block for ResNet-style architectures

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Stride of the convolution
    norm : str, optional
        Normalization method. One of "group", "batch", "instance", or None
    activation : str, optional
        Activation function. One of "relu", "leakyrelu", or None
    """

    @configurable
    def __init__(self, in_channels, out_channels, stride=1, norm='group', activation='relu'):
        super(BottleneckBlock, self).__init__()
        if norm is not None:
            if norm.lower() == 'group':
                num_groups = out_channels // 8
                norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels // 4)
                norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels // 4)
                norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
                if not stride == 1:
                    norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            elif norm.lower() == 'batch':
                norm1 = nn.BatchNorm2d(out_channels // 4)
                norm2 = nn.BatchNorm2d(out_channels // 4)
                norm3 = nn.BatchNorm2d(out_channels)
                if not stride == 1:
                    norm4 = nn.BatchNorm2d(out_channels)
            elif norm.lower() == 'instance':
                norm1 = nn.InstanceNorm2d(out_channels // 4)
                norm2 = nn.InstanceNorm2d(out_channels // 4)
                norm3 = nn.InstanceNorm2d(out_channels)
                if not stride == 1:
                    norm4 = nn.InstanceNorm2d(out_channels)
            elif norm.lower() == 'none':
                norm1 = nn.Identity()
                norm2 = nn.Identity()
                norm3 = nn.Identity()
                if not stride == 1:
                    norm4 = nn.Identity()
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()
            norm3 = nn.Identity()
            if not stride == 1:
                norm4 = nn.Identity()
        if activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.residual_fn = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1), norm1, self.activation, nn.Conv2d(out_channels // 4, out_channels // 4, stride=stride, kernel_size=3, padding=1), norm2, self.activation, nn.Conv2d(out_channels // 4, out_channels, kernel_size=1), norm3)
        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), norm4)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS, 'stride': cfg.STRIDE, 'norm': cfg.NORM, 'activation': cfg.ACTIVATION}

    def forward(self, x):
        out = self.residual_fn(x)
        out = self.activation(out + self.shortcut(x))
        return out


class BottleneckEncoder(nn.Module):
    """
    ResNet-style encoder with bottleneck residual blocks

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    norm : str
        Normalization layer to use. One of "batch", "instance", "group", or None
    p_dropout : float
        Dropout probability
    layer_config : list of int or tuple of int
        Configuration of encoder's layers
    intermediate_features : bool
        Whether to return intermediate features to get a feature hierarchy
    """

    @configurable
    def __init__(self, in_channels=3, out_channels=128, norm='batch', p_dropout=0.0, layer_config=(32, 64, 96), intermediate_features=False):
        super(BottleneckEncoder, self).__init__()
        self.intermediate_features = intermediate_features
        norm = norm.lower()
        assert norm in ('group', 'batch', 'instance', 'none')
        start_channels = layer_config[0]
        if norm == 'group':
            norm_fn = nn.GroupNorm(num_groups=8, num_channels=start_channels)
        elif norm == 'batch':
            norm_fn = nn.BatchNorm2d(start_channels)
        elif norm == 'instance':
            norm_fn = nn.InstanceNorm2d(start_channels)
        elif norm == 'none':
            norm_fn = nn.Identity()
        layers = nn.ModuleList([nn.Conv2d(in_channels, start_channels, kernel_size=7, stride=2, padding=3), norm_fn, nn.ReLU(inplace=True)])
        for i in range(len(layer_config)):
            if i == 0:
                stride = 1
            else:
                stride = 2
            layers.append(self._make_layer(start_channels, layer_config[i], stride, norm))
            start_channels = layer_config[i]
        layers.append(nn.Conv2d(layer_config[-1], out_channels, kernel_size=1))
        dropout = nn.Identity()
        if self.training and p_dropout > 0:
            dropout = nn.Dropout2d(p=p_dropout)
        layers.append(dropout)
        self.encoder = layers
        if self.intermediate_features is False:
            self.encoder = nn.Sequential(*self.encoder)
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, norm):
        layer1 = BottleneckBlock(in_channels, out_channels, stride, norm)
        layer2 = BottleneckBlock(out_channels, out_channels, stride=1, norm=norm)
        return nn.Sequential(*[layer1, layer2])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        return {'in_channels': cfg.IN_CHANNELS, 'out_channels': cfg.OUT_CHANNELS, 'norm': cfg.NORM, 'p_dropout': cfg.P_DROPOUT, 'layer_config': cfg.LAYER_CONFIG, 'intermediate_features': cfg.INTERMEDIATE_FEATURES}

    def forward(self, x):
        if self.intermediate_features is True:
            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)
                if isinstance(self.encoder[i], nn.Sequential):
                    features.append(x)
            features.append(x)
            return features
        else:
            is_list = isinstance(x, tuple) or isinstance(x, list)
            if is_list:
                batch_dim = x[0].shape[0]
                x = torch.cat(x, dim=0)
            out = self.encoder(x)
            if is_list:
                out = torch.split(out, [batch_dim, batch_dim], dim=0)
            return out


FUNCTIONAL_REGISTRY = Registry('FUNCTIONAL')


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for optical flow estimation.
    Used in **DICL** (https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html)

    Parameters
    ----------
    norm : str, default: "l1"
        The norm to use for the loss. Can be either "l2", "l1" or "robust"
    q : float, default: 0.4
        This parameter is used in robust loss for fine tuning. q < 1 gives less penalty to outliers
    eps : float, default: 0.01
        This parameter is a small constant used in robust loss to stabilize fine tuning.
    weights : list
        The weights to use for each scale
    average : str, default: "mean"
        The mode to set the average of the EPE map.
        If "mean", the mean of the EPE map is returned.
        If "sum", the EPE map is summed and divided by the batch size.
    resize_flow : str, default: "upsample"
        The mode to resize flow.
        If "upsample", predicted flow will be upsampled to the size of the ground truth.
        If "downsample", ground truth flow will be downsampled to the size of the predicted flow.
    extra_mask : torch.Tensor
        A mask to apply to the loss. Useful for removing the loss on the background
    use_valid_range : bool
        Whether to use the valid range of flow values for the loss
    valid_range : list
        The valid range of flow values for each scale
    """

    @configurable
    def __init__(self, norm='l1', q=0.4, eps=0.01, weights=(1, 0.5, 0.25), average='mean', resize_flow='upsample', extra_mask=None, use_valid_range=True, valid_range=None):
        super(MultiScaleLoss, self).__init__()
        assert norm.lower() in ('l1', 'l2', 'robust'), 'Norm must be one of L1, L2, Robust'
        assert resize_flow.lower() in ('upsample', 'downsample'), 'Resize flow must be one of upsample or downsample'
        assert average.lower() in ('mean', 'sum'), 'Average must be one of mean or sum'
        self.norm = norm.lower()
        self.q = q
        self.eps = eps
        self.weights = weights
        self.extra_mask = extra_mask
        self.use_valid_range = use_valid_range
        self.valid_range = valid_range
        self.average = average.lower()
        self.resize_flow = resize_flow.lower()

    @classmethod
    def from_config(cls, cfg):
        return {'norm': cfg.NORM, 'weights': cfg.WEIGHTS, 'average': cfg.AVERAGE, 'resize_flow': cfg.RESIZE_FLOW, 'extra_mask': cfg.EXTRA_MASK, 'use_valid_range': cfg.USE_VALID_RANGE, 'valid_range': cfg.VALID_RANGE}

    def forward(self, pred, label):
        if label.shape[1] == 3:
            """Ignore valid mask for Multiscale Loss."""
            mask = label[:, 2:, :, :]
            label = label[:, :2, :, :]
        loss = 0
        b, c, h, w = label.size()
        if type(pred) is not tuple and type(pred) is not list and type(pred) is not set:
            pred = {pred}
        for i, level_pred in enumerate(pred):
            if self.resize_flow.lower() == 'upsample':
                real_flow = F.interpolate(level_pred, (h, w), mode='bilinear', align_corners=True)
                real_flow[:, 0, :, :] = real_flow[:, 0, :, :] * (w / level_pred.shape[3])
                real_flow[:, 1, :, :] = real_flow[:, 1, :, :] * (h / level_pred.shape[2])
                target = label
            elif self.resize_flow.lower() == 'downsample':
                b, c, h, w = level_pred.shape
                target = F.adaptive_avg_pool2d(label, [h, w])
                real_flow = level_pred
            if self.norm == 'l2':
                loss_value = torch.norm(real_flow - target, p=2, dim=1)
            elif self.norm == 'robust':
                loss_value = torch.norm(real_flow - target, p=1, dim=1)
                loss_value = (loss_value + self.eps) ** self.q
            elif self.norm == 'l1':
                loss_value = torch.norm(real_flow - target, p=1, dim=1)
            if self.use_valid_range and self.valid_range is not None:
                with torch.no_grad():
                    mask = (target[:, 0, :, :].abs() <= self.valid_range[i][1]) & (target[:, 1, :, :].abs() <= self.valid_range[i][0])
            else:
                with torch.no_grad():
                    mask = torch.ones(target[:, 0, :, :].shape).type_as(target)
            loss_value = loss_value * mask.float()
            if self.extra_mask is not None:
                val = self.extra_mask > 0
                loss_value = loss_value[val]
            if self.average.lower() == 'mean':
                level_loss = loss_value.mean() * self.weights[i]
            elif self.average.lower() == 'sum':
                level_loss = loss_value.sum() / b * self.weights[i]
            loss += level_loss
        loss = loss / len(pred)
        return loss


class SequenceLoss(nn.Module):
    """
    Sequence loss for optical flow estimation.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    gamma : float
        Weight for the loss
    max_flow : float
        Maximum flow magnitude
    """

    @configurable
    def __init__(self, gamma=0.8, max_flow=400):
        super(SequenceLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    @classmethod
    def from_config(cls, cfg):
        return {'gamma': cfg.GAMMA, 'max_flow': cfg.MAX_FLOW}

    def forward(self, pred, label):
        assert label.shape[1] == 3, 'Incorrect channel dimension. Set append valid mask to True in DataloaderCreator to append the valid data mask in the target label.'
        n_preds = len(pred)
        flow_loss = 0.0
        flow, valid = label[:, :2, :, :], label[:, 2:, :, :]
        valid = torch.squeeze(valid, dim=1)
        mag = torch.sqrt(torch.sum(flow ** 2, dim=1))
        valid = (valid >= 0.5) & (mag < self.max_flow)
        for i in range(n_preds):
            i_weight = self.gamma ** (n_preds - i - 1)
            i_loss = torch.abs(pred[i] - flow)
            flow_loss += i_weight * torch.mean(valid[:, None] * i_loss)
        return flow_loss


class BaseModule(nn.Module):
    """
    A wrapper for torch.nn.Module to maintain common
    module functionalities.

    """

    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self):
        pass

    def freeze_batch_norm(self):
        """
        Set Batch Norm layers to evaluation state.
        This method can be used for fine tuning.

        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class DisplacementAwareProjection(nn.Module):
    """
    Displacement-aware projection layer

    Parameters
    ----------
    max_displacement : int, optional
        Maximum displacement
    temperature : bool, optional
        If True, use temperature scaling
    temp_factor : float, optional
        Temperature scaling factor
    """

    @configurable
    def __init__(self, max_displacement=3, temperature=False, temp_factor=1e-06):
        super(DisplacementAwareProjection, self).__init__()
        self.temperature = temperature
        self.temp_factor = temp_factor
        dim_c = (2 * max_displacement + 1) ** 2
        if self.temperature:
            self.dap_layer = ConvNormRelu(dim_c, 1, kernel_size=1, padding=0, stride=1, norm=None, activation=None)
        else:
            self.dap_layer = ConvNormRelu(dim_c, dim_c, kernel_size=1, padding=0, stride=1, norm=None, activation=None)

    @classmethod
    def from_config(cls, cfg):
        return {'max_displacement': cfg.MAX_DISPLACEMENT, 'temperature': cfg.TEMPERATURE, 'temp_factor': cfg.TEMP_FACTOR}

    def forward(self, x):
        x = x.squeeze(1)
        bs, du, dv, h, w = x.shape
        x = x.view(bs, du * dv, h, w)
        if self.temperature:
            temp = self.dap_layer(x) + self.temp_factor
            x = x * temp
        else:
            x = self.dap_layer(x)
        return x.view(bs, du, dv, h, w).unsqueeze(1)


SIMILARITY_REGISTRY = Registry('SIMILARITY')


class CorrelationLayer(nn.Module):
    """
    Correlation layer in pure PyTorch
    Only supports specific values of the following parameters:
    - kernel_size: 1
    - stride_1: 1
    - stride_2: 1
    - corr_multiply: 1

    Parameters
    ----------
    pad_size : int
        Padding size for the correlation layer
    max_displacement : int
        Maximum displacement for the correlation computation
    """

    @configurable
    def __init__(self, pad_size=4, max_displacement=4):
        super().__init__()
        self.max_h_disp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    @classmethod
    def from_config(cls, cfg):
        return {'pad_size': cfg.PAD_SIZE, 'max_displacement': cfg.MAX_DISPLACEMENT}

    def forward(self, features1, features2):
        features2_pad = self.padlayer(features2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_h_disp + 1), torch.arange(0, 2 * self.max_h_disp + 1)])
        H, W = features1.shape[2], features1.shape[3]
        output = torch.cat([torch.mean(features1 * features2_pad[:, :, dy:dy + H, dx:dx + W], 1, keepdim=True) for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return output


def iter_spatial_correlation_sample(input1: torch.Tensor, input2: torch.Tensor, kernel_size: Union[int, Tuple[int, int]]=1, patch_size: Union[int, Tuple[int, int]]=1, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, dilation_patch: Union[int, Tuple[int, int]]=1) ->torch.Tensor:
    """Apply spatial correlation sampling from input1 to input2 using iteration in PyTorch.
    This docstring is taken and adapted from the original package.
    Every parameter except input1 and input2 can be either single int or a pair of int. For more information about
    Spatial Correlation Sampling, see this page. https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Parameters
    ----------
    input1 : torch.Tensor
        The origin feature map.
    input2 : torch.Tensor
        The target feature map.
    kernel_size : Union[int, Tuple[int, int]], default 1
        Total size of your correlation kernel, in pixels
    patch_size : Union[int, Tuple[int, int]], default 1
        Total size of your patch, determining how many different shifts will be applied.
    stride : Union[int, Tuple[int, int]], default 1
        Stride of the spatial sampler, will modify output height and width.
    padding : Union[int, Tuple[int, int]], default 0
        Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
    dilation : Union[int, Tuple[int, int]], default 1
        Similar to dilation in convolution.
    dilation_patch : Union[int, Tuple[int, int]], default 1
        Step for every shift in patch.

    Returns
    -------
    torch.Tensor
        Result of correlation sampling.
    Raises
    ------
    NotImplementedError
        If kernel_size != 1.
    NotImplementedError
        If dilation != 1.
    """
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (dilation_patch, dilation_patch) if isinstance(dilation_patch, int) else dilation_patch
    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError('Only kernel_size=1 is supported.')
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError('Only dilation=1 is supported.')
    if patch_size[0] % 2 == 0 or patch_size[1] % 2 == 0:
        raise NotImplementedError('Only odd patch sizes are supperted.')
    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))
    max_displacement = dilation_patch[0] * (patch_size[0] - 1) // 2, dilation_patch[1] * (patch_size[1] - 1) // 2
    input2 = F.pad(input2, (max_displacement[1], max_displacement[1], max_displacement[0], max_displacement[0]))
    b, _, h, w = input1.shape
    input1 = input1[:, :, ::stride[0], ::stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw)
    for i in range(0, 2 * max_displacement[0] + 1, dilation_patch[0]):
        for j in range(0, 2 * max_displacement[1] + 1, dilation_patch[1]):
            p2 = input2[:, :, i:i + h, j:j + w]
            p2 = p2[:, :, ::stride[0], ::stride[1]]
            corr[:, i // dilation_patch[0], j // dilation_patch[1]] = (input1 * p2).sum(dim=1)
    return corr


class IterSpatialCorrelationSampler(nn.Module):
    """
    Spatial correlation sampling from two inputs using iteration in PyTorch

    Parameters
    ----------
    kernel_size : Union[int, Tuple[int, int]], default 1
        Total size of your correlation kernel, in pixels
    patch_size : Union[int, Tuple[int, int]], default 1
        Total size of your patch, determining how many different shifts will be applied.
    stride : Union[int, Tuple[int, int]], default 1
        Stride of the spatial sampler, will modify output height and width.
    padding : Union[int, Tuple[int, int]], default 0
        Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
    dilation : Union[int, Tuple[int, int]], default 1
        Similar to dilation in convolution.
    dilation_patch : Union[int, Tuple[int, int]], default 1
        Step for every shift in patch.
    """

    @configurable
    def __init__(self, kernel_size: Union[int, Tuple[int, int]]=1, patch_size: Union[int, Tuple[int, int]]=1, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, dilation_patch: Union[int, Tuple[int, int]]=1) ->None:
        super(IterSpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    @classmethod
    def from_config(cls, cfg):
        return {'kernel_size': cfg.KERNEL_SIZE, 'patch_size': cfg.PATCH_SIZE, 'stride': cfg.STRIDE, 'padding': cfg.PADDING, 'dilation': cfg.DILATION, 'dilation_patch': cfg.DILATION_PATCH}

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) ->torch.Tensor:
        """
        Compute the correlation sampling from input1 to input2

        Parameters
        ----------
        input1 : torch.Tensor
            The origin feature map.
        input2 : torch.Tensor
            The target feature map.

        Returns
        -------
        torch.Tensor
            Result of correlation sampling

        """
        return iter_spatial_correlation_sample(input1=input1, input2=input2, kernel_size=self.kernel_size, patch_size=self.patch_size, stride=self.stride, padding=self.padding, dilation=self.dilation, dilation_patch=self.dilation_patch)


class Conv2DMatching(nn.Module):
    """
    Convolutional matching/filtering network for cost volume learning

    Parameters
    ----------
    config : tuple of int or list of int
        Configuration of the convolutional layers in the network
    """

    @configurable
    def __init__(self, config=(64, 96, 128, 64, 32, 1)):
        super(Conv2DMatching, self).__init__()
        self.matching_net = nn.Sequential(ConvNormRelu(config[0], config[1], kernel_size=3, padding=1, dilation=1), ConvNormRelu(config[1], config[2], kernel_size=3, stride=2, padding=1), ConvNormRelu(config[2], config[2], kernel_size=3, padding=1, dilation=1), ConvNormRelu(config[2], config[3], kernel_size=3, padding=1, dilation=1), ConvNormRelu(config[3], config[4], kernel_size=4, padding=1, stride=2, deconv=True), nn.Conv2d(config[4], config[5], kernel_size=3, stride=1, padding=1, bias=True))

    @classmethod
    def from_config(cls, cfg):
        return {'config': cfg.CONFIG}

    def forward(self, x):
        x = self.matching_net(x)
        return x


class Custom2DConvMatching(nn.Module):
    """
    Convolutional matching/filtering network for cost volume learning with custom convolutions

    Parameters
    ----------
    config : tuple of int or list of int
        Configuration of the convolutional layers in the network
    kernel_size : int
        Kernel size of the convolutional layers
    **kwargs
        Additional keyword arguments for the convolutional layers
    """

    @configurable
    def __init__(self, config=(16, 32, 16, 1), kernel_size=3, **kwargs):
        super(Custom2DConvMatching, self).__init__()
        matching_net = nn.ModuleList()
        for i in range(len(config) - 2):
            matching_net.append(ConvNormRelu(config[i], config[i + 1], kernel_size=kernel_size, **kwargs))
        matching_net.append(nn.Conv2d(config[-2], config[-1], kernel_size=1))
        self.matching_net = nn.Sequential(*matching_net)

    @classmethod
    def from_config(cls, cfg):
        return {'config': cfg.CONFIG, 'kernel_size': cfg.KERNEL_SIZE}

    def forward(self, x):
        x = self.matching_net(x)
        return x


class LearnableMatchingCost(nn.Module):
    """
    Learnable matching cost network for cost volume learning. Used in **DICL** (https://arxiv.org/abs/2010.14851)

    Parameters
    ----------
    max_u : int, optional
        Maximum displacement in the horizontal direction
    max_v : int, optional
        Maximum displacement in the vertical direction
    config : tuple of int or list of int, optional
        Configuration of the convolutional layers (matching net) in the network
    remove_warp_hole : bool, optional
        Whether to remove the warp holes in the cost volume
    cuda_cost_compute : bool, optional
        Whether to compute the cost volume on the GPU
    matching_net : Optional[nn.Module], optional
        Custom matching network, by default None, which uses a Conv2DMatching network
    """

    @configurable
    def __init__(self, max_u=3, max_v=3, config=(64, 96, 128, 64, 32, 1), remove_warp_hole=True, cuda_cost_compute=False, matching_net=None):
        super(LearnableMatchingCost, self).__init__()
        if matching_net is not None:
            self.matching_net = matching_net
        else:
            self.matching_net = Conv2DMatching(config=config)
        self.max_u = max_u
        self.max_v = max_v
        self.remove_warp_hole = remove_warp_hole
        self.cuda_cost_compute = cuda_cost_compute

    @classmethod
    def from_config(cls, cfg):
        return {'max_u': cfg.MAX_U, 'max_v': cfg.MAX_V, 'config': cfg.CONFIG, 'remove_warp_hole': cfg.REMOVE_WARP_HOLE}

    def forward(self, x, y):
        size_u = 2 * self.max_u + 1
        size_v = 2 * self.max_v + 1
        _, c, height, width = x.shape
        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], 2 * c, 2 * self.max_u + 1, 2 * self.max_v + 1, height, width).zero_()
        if self.cuda_cost_compute:
            corr = SpatialCorrelationSampler(kernel_size=1, patch_size=(int(1 + 2 * 3), int(1 + 2 * 3)), stride=1, padding=0, dilation_patch=1)
            cost = corr(x, y)
        else:
            for i in range(2 * self.max_u + 1):
                ind = i - self.max_u
                for j in range(2 * self.max_v + 1):
                    indd = j - self.max_v
                    cost[:, :c, i, j, max(0, -indd):height - indd, max(0, -ind):width - ind] = x[:, :, max(0, -indd):height - indd, max(0, -ind):width - ind]
                    cost[:, c:, i, j, max(0, -indd):height - indd, max(0, -ind):width - ind] = y[:, :, max(0, +indd):height + indd, max(0, ind):width + ind]
        if self.remove_warp_hole:
            valid_mask = cost[:, c:, ...].sum(dim=1) != 0
            valid_mask = valid_mask.detach()
            cost = cost * valid_mask.unsqueeze(1).float()
        cost = cost.permute([0, 2, 3, 1, 4, 5]).contiguous()
        cost = cost.view(x.size()[0] * size_u * size_v, c * 2, x.size()[2], x.size()[3])
        cost = self.matching_net(cost)
        cost = cost.view(x.size()[0], size_u, size_v, 1, x.size()[2], x.size()[3])
        cost = cost.permute([0, 3, 1, 2, 4, 5]).contiguous()
        return cost


class MockOpticalFlowModel(BaseModule):

    def __init__(self, img_channels):
        super().__init__()
        self.model = nn.Conv2d(img_channels * 2, 2, kernel_size=1)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=-3)
        mock_flow_prediction = self.model(x)
        flow_up = F.interpolate(mock_flow_prediction, img1.shape[-2:], mode='bilinear', align_corners=True)
        output = {'flow_preds': [mock_flow_prediction], 'flow_upsampled': flow_up}
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModule,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (ConvNormRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureProjection4D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     True),
    (FlowEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (FlowHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (MockOpticalFlowModel,
     lambda: ([], {'img_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MotionEncoder,
     lambda: ([], {'corr_radius': 4, 'corr_levels': 4}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {}),
     True),
    (PyramidPooling,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmallMotionEncoder,
     lambda: ([], {'corr_radius': 4, 'corr_levels': 4}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])], {}),
     True),
]

class Test_neu_vi_ezflow(_paritybench_base):
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

