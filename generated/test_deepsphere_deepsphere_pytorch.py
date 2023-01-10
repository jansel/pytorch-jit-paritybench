import sys
_module = sys.modules[__name__]
del sys
deepsphere = _module
data = _module
datasets = _module
dataset = _module
transforms = _module
transforms = _module
layers = _module
chebyshev = _module
samplings = _module
equiangular_pool_unpool = _module
healpix_pool_unpool = _module
icosahedron_pool_unpool = _module
models = _module
spherical_unet = _module
decoder = _module
encoder = _module
unet_model = _module
utils = _module
tests = _module
test_foo = _module
initialization = _module
laplacian_funcs = _module
parser = _module
samplings = _module
stats_extractor = _module
conf = _module
scripts = _module
run_ar_tc = _module
temporality = _module
run_ar_tc = _module
setup = _module

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


import itertools


import numpy as np


from torch.utils.data import Dataset


from torchvision.datasets.utils import download_and_extract_archive


import torch


import math


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torchvision import transforms


from scipy import sparse


from scipy.sparse import coo_matrix


from sklearn.model_selection import train_test_split


from torch import optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V, Fin = inputs.shape
    K, Fin, Fout = weight.shape
    x0 = inputs.permute(1, 2, 0).contiguous()
    x0 = x0.view([V, Fin * B])
    inputs = x0.unsqueeze(0)
    if K > 0:
        x1 = torch.sparse.mm(laplacian, x0)
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)
        for _ in range(1, K - 1):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)
            x0, x1 = x1, x2
    inputs = inputs.view([K, V, Fin, B])
    inputs = inputs.permute(3, 1, 2, 0).contiguous()
    inputs = inputs.view([B * V, Fin * K])
    weight = weight.view(Fin * K, Fout)
    inputs = inputs.matmul(weight)
    inputs = inputs.view([B, V, Fout])
    return inputs


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, conv=cheb_conv):
        """Initialize the Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv
        shape = kernel_size, in_channels, out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.

        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs


class SphericalChebConv(nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_size (int): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.register_buffer('laplacian', lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size)

    def state_dict(self, *args, **kwargs):
        """! WARNING !

        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith('laplacian'):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def forward(self, x):
        """Forward pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.chebconv(self.laplacian, x)
        return x


def equiangular_bandwidth(nodes):
    """Calculate the equiangular bandwidth based on input nodes

    Args:
        nodes (int): the number of nodes should be a power of 4

    Returns:
        int: the corresponding bandwidth
    """
    bw = math.sqrt(nodes) / 2
    return bw


def equiangular_dimension_unpack(nodes, ratio):
    """Calculate the two underlying dimensions
    from the total number of nodes

    Args:
        nodes (int): combined dimensions
        ratio (float): ratio between the two dimensions

    Returns:
        int, int: separated dimensions
    """
    dim1 = int((nodes / ratio) ** 0.5)
    dim2 = int((nodes * ratio) ** 0.5)
    return dim1, dim2


def equiangular_calculator(tensor, ratio):
    """From a 3D input tensor and a known ratio between the latitude
    dimension and longitude dimension of the data, reformat the 3D input
    into a 4D output while also obtaining the bandwidth.

    Args:
        tensor (:obj:`torch.tensor`): 3D input tensor
        ratio (float): the ratio between the latitude and longitude dimension of the data

    Returns:
        :obj:`torch.tensor`, int, int: 4D tensor, the bandwidths for lat. and long.
    """
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    bw_dim1 = equiangular_bandwidth(dim1)
    bw_dim2 = equiangular_bandwidth(dim2)
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor, [bw_dim1, bw_dim2]


def reformat(x):
    """Reformat the input from a 4D tensor to a 3D tensor

    Args:
        x (:obj:`torch.tensor`): a 4D tensor
    Returns:
        :obj:`torch.tensor`: a 3D tensor
    """
    x = x.permute(0, 2, 3, 1)
    N, D1, D2, Feat = x.size()
    x = x.view(N, D1 * D2, Feat)
    return x


class EquiangularMaxPool(nn.MaxPool1d):
    """EquiAngular Maxpooling module using MaxPool 1d from torch
    """

    def __init__(self, ratio, return_indices=False):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """calls Maxpool1d and if desired, keeps indices of the pixels pooled to unpool them

        Args:
            input (:obj:`torch.tensor`): batch x pixels x features

        Returns:
            tuple(:obj:`torch.tensor`, list(int)): batch x pooled pixels x features and the indices of the pixels pooled
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        if self.return_indices:
            x, indices = F.max_pool2d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool2d(x, self.kernel_size)
        x = reformat(x)
        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output


class EquiangularAvgPool(nn.AvgPool1d):
    """EquiAngular Average Pooling using Average Pooling 1d from pytorch
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4)

    def forward(self, x):
        """calls Avgpool1d

        Args:
            x (:obj:`torch.tensor`): batch x pixels x features

        Returns:
            :obj:`torch.tensor` -- batch x pooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.kernel_size)
        x = reformat(x)
        return x


class EquiangularMaxUnpool(nn.MaxUnpool1d):
    """Equiangular Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by EquiAngMaxPool

        Args:
            x (:obj:`torch.tensor`): batch x pixels x features
            indices (int): indices of pixels equiangular maxpooled previously

        Returns:
            :obj:`torch.tensor`: batch x unpooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.max_unpool2d(x, indices, kernel_size=(4, 4))
        x = reformat(x)
        return x


class EquiangularAvgUnpool(nn.Module):
    """EquiAngular Average Unpooling version 1 using the interpolate function when unpooling
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        self.kernel_size = 4
        super().__init__()

    def forward(self, x):
        """calls pytorch's interpolate function to create the values while unpooling based on the nearby values
        Args:
            x (:obj:`torch.tensor`): batch x pixels x features
        Returns:
            :obj:`torch.tensor`: batch x unpooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.kernel_size), mode='nearest')
        x = reformat(x)
        return x


class HealpixMaxPool(nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self, return_indices=False):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch

        Args:
            x (:obj:`torch.tensor`):[batch x pixels x features]

        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [batch x pooled pixels x features] and indices of pooled pixels
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size)
        else:
            x = F.max_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)
        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output


class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch

        Arguments:
            x (:obj:`torch.tensor`): [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """Healpix Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by HealpixMaxPool

        Args:
            tuple(x (:obj:`torch.tensor`) : [batch x pixels x features]
            indices (int)): indices of pixels equiangular maxpooled previously

        Returns:
            [:obj:`torch.tensor`] -- [batch x unpooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.max_unpool1d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x


class HealpixAvgUnpool(nn.Module):
    """Healpix Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        self.kernel_size = 4
        super().__init__()

    def forward(self, x):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor

        Arguments:
            x (:obj:`torch.tensor`): [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x unpooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, scale_factor=self.kernel_size, mode='nearest')
        x = x.permute(0, 2, 1)
        return x


class IcosahedronPool(nn.Module):
    """Isocahedron Pooling, consists in keeping only a subset of the original pixels (considering the ordering of an isocahedron sampling method).
    """

    def forward(self, x):
        """Forward function calculates the subset of pixels to keep based on input size and the kernel_size.

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pixels pooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        pool_order = order - 1
        subset_pixels_keep = int(10 * math.pow(4, pool_order) + 2)
        return x[:, :subset_pixels_keep, :]


class IcosahedronUnpool(nn.Module):
    """Isocahedron Unpooling, consists in adding 1 values to match the desired un pooling size
    """

    def forward(self, x):
        """Forward calculates the subset of pixels that will result from the unpooling kernel_size and then adds 1 valued pixels to match this size

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x pixels unpooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        unpool_order = order + 1
        additional_pixels = int(10 * math.pow(4, unpool_order) + 2)
        subset_pixels_add = additional_pixels - M
        return F.pad(x, (0, 0, 0, subset_pixels_add, 0, 0), 'constant', value=1)


class SphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = F.relu(x.permute(0, 2, 1))
        return x


class SphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb_bn = SphericalChebBN(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb_bn(x)
        return x


class SphericalChebBNPoolCheb(nn.Module):
    """Building Block calling a SphericalChebBNPool block then a SphericalCheb.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalChebConv(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x


class SphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = SphericalChebBN(in_channels + out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = torch.cat((x, concat_data), dim=2)
        x = self.spherical_cheb_bn(x)
        return x


class Decoder(nn.Module):
    """The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.dec_l1 = SphericalChebBNPoolConcat(512, 512, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = SphericalChebBNPoolConcat(512, 256, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = SphericalChebBNPoolConcat(256, 128, laps[3], self.unpooling, self.kernel_size)
        self.dec_l4 = SphericalChebBNPoolConcat(128, 64, laps[4], self.unpooling, self.kernel_size)
        self.dec_l5 = SphericalChebBNPoolCheb(64, 32, 3, laps[5], self.unpooling, self.kernel_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        x = self.dec_l4(x, x_enc4)
        x = self.dec_l5(x)
        if not self.training:
            x = self.softmax(x)
        return x


class SphericalChebBN2(nn.Module):
    """Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spherical_cheb_bn_1 = SphericalChebBN(in_channels, middle_channels, lap, kernel_size)
        self.spherical_cheb_bn_2 = SphericalChebBN(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x


class SphericalChebPool(nn.Module):
    """Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        return x


class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.enc_l5 = SphericalChebBN2(16, 32, 64, laps[5], self.kernel_size)
        self.enc_l4 = SphericalChebBNPool(64, 128, laps[4], self.pooling, self.kernel_size)
        self.enc_l3 = SphericalChebBNPool(128, 256, laps[3], self.pooling, self.kernel_size)
        self.enc_l2 = SphericalChebBNPool(256, 512, laps[2], self.pooling, self.kernel_size)
        self.enc_l1 = SphericalChebBNPool(512, 512, laps[1], self.pooling, self.kernel_size)
        self.enc_l0 = SphericalChebPool(512, 512, laps[0], self.pooling, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x vertices x channels/features]
        """
        x_enc5 = self.enc_l5(x)
        x_enc4 = self.enc_l4(x_enc5)
        x_enc3 = self.enc_l3(x_enc4)
        x_enc2 = self.enc_l2(x_enc3)
        x_enc1 = self.enc_l1(x_enc2)
        x_enc0 = self.enc_l0(x_enc1)
        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4


class EncoderTemporalConv(Encoder):
    """Encoder for the Spherical UNet temporality with convolution.
    """

    def __init__(self, pooling, laps, sequence_length, kernel_size):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            sequence_length (int): The number of images used per sample.
            kernel_size (int): Polynomial degree.
        """
        super().__init__(pooling, laps, kernel_size)
        self.sequence_length = sequence_length
        self.enc_l5 = SphericalChebBN2(self.enc_l5.in_channels * self.sequence_length, self.enc_l5.in_channels * self.sequence_length, self.enc_l5.out_channels, laps[5], self.kernel_size)


class Equiangular:
    """Equiangular class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, ratio=1, mode='average'):
        """Initialize equiangular pooling and unpooling objects.

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == 'max':
            self.__pooling = EquiangularMaxPool(ratio)
            self.__unpooling = EquiangularMaxUnpool(ratio)
        else:
            self.__pooling = EquiangularAvgPool(ratio)
            self.__unpooling = EquiangularAvgUnpool(ratio)

    @property
    def pooling(self):
        """Getter for the pooling class
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Getter for the unpooling class
        """
        return self.__unpooling


class Healpix:
    """Healpix class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode='average'):
        """Initialize healpix pooling and unpooling objects.

        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == 'max':
            self.__pooling = HealpixMaxPool()
            self.__unpooling = HealpixMaxUnpool()
        else:
            self.__pooling = HealpixAvgPool()
            self.__unpooling = HealpixAvgUnpool()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling


class Icosahedron:
    """Icosahedron class, which simply groups together the corresponding pooling and unpooling.
    """

    def __init__(self):
        """Initialize icosahedron pooling and unpooling objects.
        """
        self.__pooling = IcosahedronPool()
        self.__unpooling = IcosahedronUnpool()

    @property
    def pooling(self):
        """Get pooling.
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling.
        """
        return self.__unpooling


def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.

    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.

    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """

    def estimate_lmax(laplacian, tol=0.005):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L
    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def get_equiangular_laplacians(nodes, depth, ratio, laplacian_type):
    """Get the equiangular laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians
    """
    laps = []
    pixel_num = nodes
    for _ in range(depth):
        dim1, dim2 = equiangular_dimension_unpack(pixel_num, ratio)
        bw1 = equiangular_bandwidth(dim1)
        bw2 = equiangular_bandwidth(dim2)
        bw = [bw1, bw2]
        G = SphereEquiangular(bandwidth=bw, sampling='SOFT')
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]


def healpix_resolution_calculator(nodes):
    """Calculate the resolution of a healpix graph
    for a given number of nodes.

    Args:
        nodes (int): number of nodes in healpix sampling

    Returns:
        int: resolution for the matching healpix graph
    """
    resolution = int(math.sqrt(nodes / 12))
    return resolution


def get_healpix_laplacians(nodes, depth, laplacian_type):
    """Get the healpix laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    for i in range(depth):
        pixel_num = nodes
        subdivisions = int(healpix_resolution_calculator(pixel_num) / 2 ** i)
        G = SphereHealpix(subdivisions, nest=True, k=20)
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]


def icosahedron_nodes_calculator(order):
    """Calculate the number of nodes
    corresponding to the order of an icosahedron graph

    Args:
        order (int): order of an icosahedron graph

    Returns:
        int: number of nodes in icosahedron sampling for that order
    """
    nodes = 10 * 4 ** order + 2
    return nodes


def icosahedron_order_calculator(nodes):
    """Calculate the order of a icosahedron graph
    for a given number of nodes.

    Args:
        nodes (int): number of nodes in icosahedron sampling

    Returns:
        int: order for the matching icosahedron graph
    """
    order = math.log((nodes - 2) / 10) / math.log(4)
    return order


def get_icosahedron_laplacians(nodes, depth, laplacian_type):
    """Get the icosahedron laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    order = icosahedron_order_calculator(nodes)
    for _ in range(depth):
        nodes = icosahedron_nodes_calculator(order)
        order_initial = icosahedron_order_calculator(nodes)
        G = SphereIcosahedron(level=int(order_initial))
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
        order -= 1
    return laps[::-1]


class SphericalUNet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        if pooling_class == 'icosahedron':
            self.pooling_class = Icosahedron()
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif pooling_class == 'healpix':
            self.pooling_class = Healpix()
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif pooling_class == 'equiangular':
            self.pooling_class = Equiangular()
            self.laps = get_equiangular_laplacians(N, depth, self.ratio, laplacian_type)
        else:
            raise ValueError('Error: sampling method unknown. Please use icosahedron, healpix or equiangular.')
        self.encoder = Encoder(self.pooling_class.pooling, self.laps, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output


class SphericalUNetTemporalLSTM(SphericalUNet):
    """Sphericall GCNN Autoencoder with LSTM.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, sequence_length, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            sequence_length (int): The number of images used per sample
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__(pooling_class, N, depth, laplacian_type, kernel_size, ratio)
        self.sequence_length = sequence_length
        n_pixels = self.laps[0].size(0)
        n_features = self.encoder.enc_l0.spherical_cheb.chebconv.in_channels
        self.lstm_l0 = nn.LSTM(input_size=n_pixels * n_features, hidden_size=n_pixels * n_features, batch_first=True)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        device = x.device
        encoders_l0 = []
        for idx in range(self.sequence_length):
            encoding = self.encoder(x[:, idx, :, :].squeeze(dim=1))
            encoders_l0.append(encoding[0].reshape(encoding[0].size(0), 1, -1))
        encoders_l0 = torch.cat(encoders_l0, axis=1)
        lstm_output_l0, _ = self.lstm_l0(encoders_l0)
        lstm_output_l0 = lstm_output_l0[:, -1, :].reshape(-1, encoding[0].size(1), encoding[0].size(2))
        output = self.decoder(lstm_output_l0, encoding[1], encoding[2], encoding[3], encoding[4])
        return output


class SphericalUNetTemporalConv(SphericalUNet):
    """Spherical GCNN Autoencoder with temporality by means of convolution over time.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, sequence_length, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            sequence_length (int): The number of images used per sample
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__(pooling_class, N, depth, laplacian_type, kernel_size, ratio)
        self.sequence_length = sequence_length
        self.encoder = EncoderTemporalConv(self.pooling_class.pooling, self.laps, self.sequence_length, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChebConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (EquiangularAvgUnpool,
     lambda: ([], {'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (HealpixAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (HealpixAvgUnpool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (HealpixMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (IcosahedronPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IcosahedronUnpool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_deepsphere_deepsphere_pytorch(_paritybench_base):
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

