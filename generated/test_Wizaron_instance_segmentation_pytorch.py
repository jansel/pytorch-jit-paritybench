import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
lib = _module
archs = _module
instance_counter = _module
modules = _module
conv_gru = _module
coord_conv = _module
recurrent_hourglass = _module
renet = _module
utils = _module
vgg16 = _module
reseg = _module
stacked_recurrent_hourglass = _module
dataset = _module
losses = _module
dice = _module
discriminative = _module
model = _module
prediction = _module
preprocess = _module
pred = _module
pred_list = _module
CVPPP = _module
data_settings = _module
model_settings = _module
training_settings = _module
settings = _module
train = _module

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


from torch import nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn.modules.loss import _Loss


from torch.nn.modules.loss import _WeightedLoss


import numpy as np


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.backends.cudnn as cudnn


class InstanceCounter(nn.Module):
    """Instance Counter Module. Basically, it is a convolutional network
    to count instances for a given feature map.

    Args:
        input_n_filters (int): Number of channels in the input image
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, 1)`

    Examples:
        >>> ins_cnt = InstanceCounter(3, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = ins_cnt(input)

        >>> ins_cnt = InstanceCounter(3, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = ins_cnt(input)
    """

    def __init__(self, input_n_filters, use_coordinates=False, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters
        self.n_filters = 32
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu
        self.__generate_cnn()
        self.output = nn.Sequential()
        self.output.add_module('linear', nn.Linear(self.n_filters, 1))
        self.output.add_module('sigmoid', nn.Sigmoid())

    def __generate_cnn(self):
        self.cnn = nn.Sequential()
        self.cnn.add_module('pool1', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv1', nn.Conv2d(self.input_n_filters, self.
            n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.cnn.add_module('relu1', nn.ReLU())
        self.cnn.add_module('conv2', nn.Conv2d(self.n_filters, self.
            n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.cnn.add_module('relu2', nn.ReLU())
        self.cnn.add_module('pool2', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv3', nn.Conv2d(self.n_filters, self.
            n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.cnn.add_module('relu3', nn.ReLU())
        self.cnn.add_module('conv4', nn.Conv2d(self.n_filters, self.
            n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.cnn.add_module('relu4', nn.ReLU())
        self.cnn.add_module('pool3', nn.AdaptiveAvgPool2d((1, 1)))
        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, with_r=True, usegpu=self.usegpu)

    def forward(self, x):
        x = self.cnn(x)
        if self.use_coordinates:
            x = x[-1]
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x


class ConvGRUCell(nn.Module):
    """Convolutional GRU Module as defined in 'Delving Deeper into
    Convolutional Networks for Learning Video Representations'
    (https://arxiv.org/pdf/1511.06432.pdf).

    Args:
        input_size (int): Number of channels in the input image
        hidden_size (int): Number of channels produced by the ConvGRU
        kernel_size (int or tuple): Size of the convolving kernel
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input:
            - `x` : `(N, C_{in}, H_{in}, W_{in})`
            - `hidden` : `(N, C_{out}, H_{in}, W_{in})` or `None`
        - Output: `next_hidden` : `(N, C_{out}, H_{in}, W_{in})`

    Examples:
        >>> n_hidden = 16
        >>> conv_gru = ConvGRUCell(3, n_hidden, 3, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> hidden = torch.rand(8, n_hidden, 64, 64)
        >>> output = conv_gru(input, None)
        >>> output = conv_gru(input, hidden)

        >>> n_hidden = 16
        >>> conv_gru = ConvGRUCell(3, n_hidden, 3, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> hidden = torch.rand(8, n_hidden, 64, 64).cuda()
        >>> output = conv_gru(input, None)
        >>> output = conv_gru(input, hidden)
    """

    def __init__(self, input_size, hidden_size, kernel_size,
        use_coordinates=False, usegpu=True):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu
        _n_inputs = self.input_size + self.hidden_size
        if self.use_coordinates:
            self.conv_gates = CoordConv(_n_inputs, 2 * self.hidden_size,
                self.kernel_size, padding=self.kernel_size // 2, with_r=
                True, usegpu=self.usegpu)
            self.conv_ct = CoordConv(_n_inputs, self.hidden_size, self.
                kernel_size, padding=self.kernel_size // 2, with_r=True,
                usegpu=self.usegpu)
        else:
            self.conv_gates = nn.Conv2d(_n_inputs, 2 * self.hidden_size,
                self.kernel_size, padding=self.kernel_size // 2)
            self.conv_ct = nn.Conv2d(_n_inputs, self.hidden_size, self.
                kernel_size, padding=self.kernel_size // 2)

    def forward(self, x, hidden):
        batch_size, _, height, width = x.size()
        if hidden is None:
            size_h = [batch_size, self.hidden_size, height, width]
            hidden = Variable(torch.zeros(size_h))
            if self.usegpu:
                hidden = hidden
        c1 = self.conv_gates(torch.cat((x, hidden), dim=1))
        rt, ut = c1.chunk(2, 1)
        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        ct = F.tanh(self.conv_ct(torch.cat((x, gated_hidden), dim=1)))
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


class AddCoordinates(object):
    """Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates(True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False, usegpu=True):
        self.with_r = with_r
        self.usegpu = usegpu

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(1).expand(
            image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(0).expand(
            image_height, image_width) / (image_width - 1.0) - 1.0
        coords = torch.stack((y_coords, x_coords), dim=0)
        if self.with_r:
            rs = (y_coords ** 2 + x_coords ** 2) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        coords = Variable(coords)
        if self.usegpu:
            coords = coords.cuda()
        image = torch.cat((coords, image), dim=1)
        return image


class CoordConv(nn.Module):
    """2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True, usegpu=False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)

        >>> coord_conv = CoordConv(3, 16, 3, with_r=True, usegpu=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, with_r=False, usegpu=True):
        super(CoordConv, self).__init__()
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)
        self.coord_adder = AddCoordinates(with_r, usegpu)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)
        return x


class CoordConvTranspose(nn.Module):
    """2D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.ConvTranspose2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True,
        >>>                                    usegpu=False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv_tr(input)

        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True,
        >>>                                    usegpu=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv_tr(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1,
        with_r=False, usegpu=True):
        super(CoordConvTranspose, self).__init__()
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, groups=groups, bias=bias, dilation=dilation)
        self.coord_adder = AddCoordinates(with_r, usegpu)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_tr_layer(x)
        return x


class CoordConvNet(nn.Module):
    """Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module adds coordinate information to inputs of each 2D convolution
    module (`torch.nn.Conv2d`).

    Assumption: ConvNet Model must contain single `Sequential` container
    (`torch.nn.modules.container.Sequential`).

    Args:
        cnn_model: A ConvNet model that must contain single `Sequential`
            container (`torch.nn.modules.container.Sequential`).
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: Same as the input of the model.
        - Output: A list that contains all outputs (including
            intermediate outputs) of the model.

    Examples:
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = cnn_model(input)

        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = cnn_model(input)
    """

    def __init__(self, cnn_model, with_r=False, usegpu=True):
        super(CoordConvNet, self).__init__()
        self.with_r = with_r
        self.cnn_model = cnn_model
        self.__get_model()
        self.__update_weights()
        self.coord_adder = AddCoordinates(self.with_r, usegpu)

    def __get_model(self):
        for module in list(self.cnn_model.modules()):
            if module.__class__ == torch.nn.modules.container.Sequential:
                self.cnn_model = module
                break

    def __update_weights(self):
        coord_channels = 2
        if self.with_r:
            coord_channels += 1
        for l in list(self.cnn_model.modules()):
            if l.__str__().startswith('Conv2d'):
                weights = l.weight.data
                out_channels, in_channels, k_height, k_width = weights.size()
                coord_weights = torch.zeros(out_channels, coord_channels,
                    k_height, k_width)
                weights = torch.cat((coord_weights, weights), dim=1)
                weights = nn.Parameter(weights)
                l.weight = weights
                l.in_channels += coord_channels

    def __get_outputs(self, x):
        outputs = []
        for layer_name, layer in self.cnn_model._modules.items():
            if layer.__str__().startswith('Conv2d'):
                x = self.coord_adder(x)
            x = layer(x)
            outputs.append(x)
        return outputs

    def forward(self, x):
        return self.__get_outputs(x)


class RecurrentHourglass(nn.Module):
    """RecurrentHourglass Module as defined in
    'Instance Segmentation and Tracking with Cosine Embeddings and Recurrent
    Hourglass Networks' (https://arxiv.org/pdf/1806.02070.pdf).

    Args:
        input_n_filters (int): Number of channels in the input image
        hidden_n_filters (int): Number of channels produced by Convolutional
            GRU module
        kernel_size (int or tuple): Size of the convolving kernels
        n_levels (int): Number of timesteps to unroll Convolutional GRU
            module
        embedding_size (int): Number of channels produced by Recurrent
            Hourglass module
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{in}, W_{in})`

    Examples:
        >>> hg = RecurrentHourglass(3, 16, 3, 5, 32, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = hg(input)

        >>> hg = RecurrentHourglass(3, 16, 3, 5, 32, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = hg(input)
    """

    def __init__(self, input_n_filters, hidden_n_filters, kernel_size,
        n_levels, embedding_size, use_coordinates=False, usegpu=True):
        super(RecurrentHourglass, self).__init__()
        assert n_levels >= 1, 'n_levels should be greater than or equal to 1.'
        self.input_n_filters = input_n_filters
        self.hidden_n_filters = hidden_n_filters
        self.kernel_size = kernel_size
        self.n_levels = n_levels
        self.embedding_size = embedding_size
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu
        self.convgru_cell = ConvGRUCell(self.hidden_n_filters, self.
            hidden_n_filters, self.kernel_size, self.use_coordinates, self.
            usegpu)
        self.__generate_pre_post_convs()

    def __generate_pre_post_convs(self):
        if self.use_coordinates:

            def __get_conv(input_n_filters, output_n_filters):
                return CoordConv(input_n_filters, output_n_filters, self.
                    kernel_size, padding=self.kernel_size // 2, with_r=True,
                    usegpu=self.usegpu)
        else:

            def __get_conv(input_n_filters, output_n_filters):
                return nn.Conv2d(input_n_filters, output_n_filters, self.
                    kernel_size, padding=self.kernel_size // 2)
        self.pre_conv_layers = [__get_conv(self.input_n_filters, self.
            hidden_n_filters)]
        for _ in range(self.n_levels - 1):
            self.pre_conv_layers.append(__get_conv(self.hidden_n_filters,
                self.hidden_n_filters))
        self.pre_conv_layers = ListModule(*self.pre_conv_layers)
        self.post_conv_layers = [__get_conv(self.hidden_n_filters, self.
            embedding_size)]
        for _ in range(self.n_levels - 1):
            self.post_conv_layers.append(__get_conv(self.hidden_n_filters,
                self.hidden_n_filters))
        self.post_conv_layers = ListModule(*self.post_conv_layers)

    def forward_encoding(self, x):
        convgru_outputs = []
        hidden = None
        for i in range(self.n_levels):
            x = F.relu(self.pre_conv_layers[i](x))
            hidden = self.convgru_cell(x, hidden)
            convgru_outputs.append(hidden)
        return convgru_outputs

    def forward_decoding(self, convgru_outputs):
        _last_conv_layer = self.post_conv_layers[self.n_levels - 1]
        _last_output = convgru_outputs[self.n_levels - 1]
        post_feature_map = F.relu(_last_conv_layer(_last_output))
        for i in range(self.n_levels - 1)[::-1]:
            post_feature_map += convgru_outputs[i]
            post_feature_map = self.post_conv_layers[i](post_feature_map)
            post_feature_map = F.relu(post_feature_map)
        return post_feature_map

    def forward(self, x):
        x = self.forward_encoding(x)
        x = self.forward_decoding(x)
        return x


class ReNet(nn.Module):
    """ReNet Module as defined in 'ReNet: A Recurrent Neural
    Network Based Alternative to Convolutional Networks'
    (https://arxiv.org/pdf/1505.00393.pdf).

    Args:
        n_input (int): Number of channels in the input image
        n_units (int): Number of channels produced by ReNet
        patch_size (tuple): Patch size in the input of ReNet
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> renet = ReNet(3, 16, (2, 2), True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = renet(input)

        >>> renet = ReNet(3, 16, (2, 2), True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = renet(input)
    """

    def __init__(self, n_input, n_units, patch_size=(1, 1), use_coordinates
        =False, usegpu=True):
        super(ReNet, self).__init__()
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])
        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1
        self.tiling = (False if self.patch_size_height == 1 and self.
            patch_size_width == 1 else True)
        if self.use_coordinates:
            self.coord_adder = AddCoordinates(with_r=True, usegpu=self.usegpu)
        rnn_hor_n_inputs = (n_input * self.patch_size_height * self.
            patch_size_width)
        if self.use_coordinates:
            rnn_hor_n_inputs += 3
        self.rnn_hor = nn.GRU(rnn_hor_n_inputs, n_units, num_layers=1,
            batch_first=True, bidirectional=True)
        self.rnn_ver = nn.GRU(n_units * 2, n_units, num_layers=1,
            batch_first=True, bidirectional=True)

    def __tile(self, x):
        if x.size(2) % self.patch_size_height == 0:
            n_height_padding = 0
        else:
            n_height_padding = self.patch_size_height - x.size(2
                ) % self.patch_size_height
        if x.size(3) % self.patch_size_width == 0:
            n_width_padding = 0
        else:
            n_width_padding = self.patch_size_width - x.size(3
                ) % self.patch_size_width
        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding
        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding
        x = F.pad(x, (n_left_padding, n_right_padding, n_top_padding,
            n_bottom_padding))
        b, n_filters, n_height, n_width = x.size()
        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0
        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width
        x = x.view(b, n_filters, new_height, self.patch_size_height,
            new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height * self.
            patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        return x

    def __swap_hw(self, x):
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous()
        return x

    def rnn_forward(self, x, hor_or_ver):
        assert hor_or_ver in ['hor', 'ver']
        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        b, n_height, n_width, n_filters = x.size()
        x = x.view(b * n_height, n_width, n_filters)
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        elif hor_or_ver == 'ver':
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)
        return x

    def forward(self, x):
        if self.tiling:
            x = self.__tile(x)
        if self.use_coordinates:
            x = self.coord_adder(x)
        x = self.rnn_forward(x, 'hor')
        x = self.rnn_forward(x, 'ver')
        return x


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class VGG16(nn.Module):
    """A module that augments VGG16 as defined in 'Very Deep Convolutional
    Networks for Large-Scale Image Recognition'
    (https://arxiv.org/pdf/1409.1556.pdf).

    1. It can return first `n_layers` of the VGG16.
    2. It can add coordinates to feature maps prior to each convolution.
    3. It can return all outputs (including intermediate outputs) of the
        VGG16.

    Args:
        n_layers (int): Use first `n_layers` layers of the VGG16
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        return_intermediate_outputs (bool, optional): If `True`, return
            outputs of the each layer in the VGG16 as a list otherwise
            return output of the last layer of first `n_layers` layers
            of the VGG16. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: Output of the last layer of the selected subpart of VGG16
            or the list that contains outputs of the each layer depending on
            `return_intermediate_outputs`

    Examples:
        >>> vgg16 = VGG16(16, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = VGG16(16, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, n_layers, pretrained=True, use_coordinates=False,
        return_intermediate_outputs=False, usegpu=True):
        super(VGG16, self).__init__()
        self.use_coordinates = use_coordinates
        self.return_intermediate_outputs = return_intermediate_outputs
        self.cnn = models.__dict__['vgg16'](pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        self.cnn = nn.Sequential(*list(self.cnn.children())[:n_layers])
        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, True, usegpu)

    def __get_outputs(self, x):
        if self.use_coordinates:
            return self.cnn(x)
        outputs = []
        for i, layer in enumerate(self.cnn.children()):
            x = layer(x)
            outputs.append(x)
        return outputs

    def forward(self, x):
        outputs = self.__get_outputs(x)
        if self.return_intermediate_outputs:
            return outputs
        return outputs[-1]


class SkipVGG16(nn.Module):
    """A module that returns output of 7th convolutional layer of the
    VGG16 along with outputs of the 2nd and 4th convolutional layers.

    Args:
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: List of outputs of the 2nd, 4th and 7th convolutional
            layers of the VGG16, respectively.

    Examples:
        >>> vgg16 = SkipVGG16(True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = SkipVGG16(True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, pretrained=True, use_coordinates=False, usegpu=True):
        super(SkipVGG16, self).__init__()
        self.use_coordinates = use_coordinates
        self.outputs = [3, 8]
        self.n_filters = [64, 128]
        self.model = VGG16(n_layers=16, pretrained=pretrained,
            use_coordinates=self.use_coordinates,
            return_intermediate_outputs=True, usegpu=usegpu)

    def forward(self, x):
        if self.use_coordinates:
            outs = self.model(x)
            out = [o for i, o in enumerate(outs) if i in self.outputs]
            out.append(outs[-1])
        else:
            out = []
            for i, layer in enumerate(list(self.model.children())[0]):
                x = layer(x)
                if i in self.outputs:
                    out.append(x)
            out.append(x)
        return out


class ReSeg(nn.Module):
    """ReSeg Module (with modifications) as defined in 'ReSeg: A Recurrent
    Neural Network-based Model for Semantic Segmentation'
    (https://arxiv.org/pdf/1511.07053.pdf).

    * VGG16 with skip Connections as base network
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

    Args:
        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in})`
            - Instance Cnt: `(N, 1)`

    Examples:
        >>> reseg = ReSeg(3, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = reseg(input)

        >>> reseg = ReSeg(3, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = reseg(input)
    """

    def __init__(self, n_classes, use_instance_seg=True, pretrained=True,
        use_coordinates=False, usegpu=True):
        super(ReSeg, self).__init__()
        self.n_classes = n_classes
        self.use_instance_seg = use_instance_seg
        self.cnn = SkipVGG16(pretrained=pretrained, use_coordinates=
            use_coordinates, usegpu=usegpu)
        self.renet1 = ReNet(256, 100, use_coordinates=use_coordinates,
            usegpu=usegpu)
        self.renet2 = ReNet(100 * 2, 100, use_coordinates=use_coordinates,
            usegpu=usegpu)
        self.upsampling1 = nn.ConvTranspose2d(100 * 2, 100, kernel_size=(2,
            2), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(100 + self.cnn.n_filters[1], 
            100, kernel_size=(2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.sem_seg_output = nn.Conv2d(100 + self.cnn.n_filters[0], self.
            n_classes, kernel_size=(1, 1), stride=(1, 1))
        if self.use_instance_seg:
            self.ins_seg_output = nn.Conv2d(100 + self.cnn.n_filters[0], 32,
                kernel_size=(1, 1), stride=(1, 1))
        self.ins_cls_cnn = InstanceCounter(100 * 2, use_coordinates, usegpu
            =usegpu)

    def forward(self, x):
        first_skip, second_skip, x_enc = self.cnn(x)
        x_enc = self.renet1(x_enc)
        x_enc = self.renet2(x_enc)
        x_dec = self.relu1(self.upsampling1(x_enc))
        x_dec = torch.cat((x_dec, second_skip), dim=1)
        x_dec = self.relu2(self.upsampling2(x_dec))
        x_dec = torch.cat((x_dec, first_skip), dim=1)
        sem_seg_out = self.sem_seg_output(x_dec)
        if self.use_instance_seg:
            ins_seg_out = self.ins_seg_output(x_dec)
        else:
            ins_seg_out = None
        ins_cls_out = self.ins_cls_cnn(x_enc)
        return sem_seg_out, ins_seg_out, ins_cls_out


class StackedRecurrentHourglass(nn.Module):
    """Stacked Recurrent Hourglass Module for instance segmentation
    as defined in 'Instance Segmentation and Tracking with Cosine
    Embeddings and Recurrent Hourglass Networks'
    (https://arxiv.org/pdf/1806.02070.pdf).

    * First four layers of VGG16
    * Two RecurrentHourglass layers
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

    Args:
        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in})`
            - Instance Cnt: `(N, 1)`

    Examples:
        >>> srhg = StackedRecurrentHourglass(4, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = srhg(input)

        >>> srhg = StackedRecurrentHourglass(4, True, True, True, True)
        >>> srhg = srhg.cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = srhg(input)
    """

    def __init__(self, n_classes, use_instance_seg=True, pretrained=True,
        use_coordinates=False, usegpu=True):
        super(StackedRecurrentHourglass, self).__init__()
        self.n_classes = n_classes
        self.use_instance_seg = use_instance_seg
        self.use_coords = use_coordinates
        self.pretrained = pretrained
        self.usegpu = usegpu
        self.base_cnn = self.__generate_base_cnn()
        self.enc_stacked_hourglass = self.__generate_enc_stacked_hg(64, 3)
        self.stacked_renet = self.__generate_stacked_renet(64, 2)
        self.decoder = self.__generate_decoder(64)
        self.semantic_seg, self.instance_seg, self.instance_count = (self.
            __generate_heads(64, 32))

    def __generate_base_cnn(self):
        base_cnn = VGG16(n_layers=4, pretrained=self.pretrained,
            use_coordinates=self.use_coords, return_intermediate_outputs=
            False, usegpu=self.usegpu)
        return base_cnn

    def __generate_enc_stacked_hg(self, input_n_filters, n_levels):
        stacked_hourglass = nn.Sequential()
        stacked_hourglass.add_module('Hourglass_1', RecurrentHourglass(
            input_n_filters=input_n_filters, hidden_n_filters=64,
            kernel_size=3, n_levels=n_levels, embedding_size=64,
            use_coordinates=self.use_coords, usegpu=self.usegpu))
        stacked_hourglass.add_module('pool_1', nn.MaxPool2d(2, stride=2))
        stacked_hourglass.add_module('Hourglass_2', RecurrentHourglass(
            input_n_filters=64, hidden_n_filters=64, kernel_size=3,
            n_levels=n_levels, embedding_size=64, use_coordinates=self.
            use_coords, usegpu=self.usegpu))
        stacked_hourglass.add_module('pool_2', nn.MaxPool2d(2, stride=2))
        return stacked_hourglass

    def __generate_stacked_renet(self, input_n_filters, n_renets):
        assert n_renets >= 1, 'n_renets should be 1 at least.'
        renet = nn.Sequential()
        renet.add_module('ReNet_1', ReNet(input_n_filters, 32, patch_size=(
            1, 1), use_coordinates=self.use_coords, usegpu=self.usegpu))
        for i in range(1, n_renets):
            renet.add_module('ReNet_{}'.format(i + 1), ReNet(32 * 2, 32,
                patch_size=(1, 1), use_coordinates=self.use_coords, usegpu=
                self.usegpu))
        return renet

    def __generate_decoder(self, input_n_filters):
        decoder = nn.Sequential()
        decoder.add_module('ConvTranspose_1', nn.ConvTranspose2d(
            input_n_filters, 64, kernel_size=(2, 2), stride=(2, 2)))
        decoder.add_module('ReLU_1', nn.ReLU())
        decoder.add_module('ConvTranspose_2', nn.ConvTranspose2d(64, 64,
            kernel_size=(2, 2), stride=(2, 2)))
        decoder.add_module('ReLU_2', nn.ReLU())
        return decoder

    def __generate_heads(self, input_n_filters, embedding_size):
        semantic_segmentation = nn.Sequential()
        semantic_segmentation.add_module('Conv_1', nn.Conv2d(
            input_n_filters, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
            )
        if self.use_instance_seg:
            instance_segmentation = nn.Sequential()
            instance_segmentation.add_module('Conv_1', nn.Conv2d(
                input_n_filters, embedding_size, kernel_size=(1, 1), stride
                =(1, 1)))
        else:
            instance_segmentation = None
        instance_counting = InstanceCounter(input_n_filters,
            use_coordinates=self.use_coords, usegpu=self.usegpu)
        return semantic_segmentation, instance_segmentation, instance_counting

    def forward(self, x):
        x = self.base_cnn(x)
        x = self.enc_stacked_hourglass(x)
        x = self.stacked_renet(x)
        x = self.decoder(x)
        sem_seg_out = self.semantic_seg(x)
        if self.use_instance_seg:
            ins_seg_out = self.instance_seg(x)
        else:
            ins_seg_out = None
        ins_count_out = self.instance_count(x)
        return sem_seg_out, ins_seg_out, ins_count_out


def dice_coefficient(input, target, smooth=1.0):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input"""
    assert input.size() == target.size(), 'Input sizes must be equal.'
    assert input.dim() == 4, 'Input must be a 4D Tensor.'
    uniques = np.unique(target.data.cpu().numpy())
    assert set(list(uniques)) <= set([0, 1]
        ), 'Target must only contain zeros and ones.'
    assert smooth > 0, 'Smooth must be greater than 0.'
    probs = F.softmax(input, dim=1)
    target_f = target.float()
    num = probs * target_f
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)
    den1 = probs * probs
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)
    den2 = target_f * target_f
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)
    dice = (2 * num + smooth) / (den1 + den2 + smooth)
    return dice


def dice_loss(input, target, optimize_bg=False, weight=None, smooth=1.0,
    size_average=True, reduce=True):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input

    weight (Variable, optional): a manual rescaling weight given to each
            class. If given, has to be a Variable of size "nclasses"""
    dice = dice_coefficient(input, target, smooth=smooth)
    if not optimize_bg:
        dice = dice[:, 1:]
    if not isinstance(weight, type(None)):
        if not optimize_bg:
            weight = weight[1:]
        weight = weight.size(0) * weight / weight.sum()
        dice = dice * weight
    dice_loss = 1 - dice.mean(1)
    if not reduce:
        return dice_loss
    if size_average:
        return dice_loss.mean()
    return dice_loss.sum()


class DiceLoss(_WeightedLoss):

    def __init__(self, optimize_bg=False, weight=None, smooth=1.0,
        size_average=True, reduce=True):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input

        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""
        super(DiceLoss, self).__init__(weight, size_average)
        self.optimize_bg = optimize_bg
        self.smooth = smooth
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return dice_loss(input, target, optimize_bg=self.optimize_bg,
            weight=self.weight, smooth=self.smooth, size_average=self.
            size_average, reduce=self.reduce)


class DiceCoefficient(torch.nn.Module):

    def __init__(self, smooth=1.0):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input"""
        super(DiceCoefficient, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        _assert_no_grad(target)
        return dice_coefficient(input, target, smooth=self.smooth)


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """means: bs, n_instances, n_filters"""
    bs, n_instances, n_filters = means.size()
    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])
        if _n_objects_sample <= 1:
            continue
        _mean_sample = means[(i), :_n_objects_sample, :]
        means_1 = _mean_sample.unsqueeze(1).expand(_n_objects_sample,
            _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)
        diff = means_1 - means_2
        _norm = torch.norm(diff, norm, 2)
        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)
        _dist_term_sample = torch.sum(torch.clamp(margin - _norm, min=0.0) ** 2
            )
        _dist_term_sample = _dist_term_sample / (_n_objects_sample * (
            _n_objects_sample - 1))
        dist_term += _dist_term_sample
    dist_term = dist_term / bs
    return dist_term


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""
    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)
    pred_repeated = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    gt_expanded = gt.unsqueeze(3)
    pred_masked = pred_repeated * gt_expanded
    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        _pred_masked_sample = pred_masked[(i), :, :_n_objects_sample]
        _gt_expanded_sample = gt_expanded[(i), :, :_n_objects_sample]
        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)
        if max_n_objects - _n_objects_sample != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)
    means = torch.stack(means)
    return means


def calculate_regularization_term(means, n_objects, norm):
    """means: bs, n_instances, n_filters"""
    bs, n_instances, n_filters = means.size()
    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[(i), :n_objects[i], :]
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs
    return reg_term


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""
    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)
    _var = torch.clamp(torch.norm(pred - means, norm, 3) - delta_v, min=0.0
        ) ** 2 * gt[:, :, :, (0)]
    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[(i), :, :n_objects[i]]
        _gt_sample = gt[(i), :, :n_objects[i], (0)]
        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs
    return var_term


def discriminative_loss(input, target, n_objects, max_n_objects, delta_v,
    delta_d, norm, usegpu):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""
    alpha = beta = 1.0
    gamma = 0.001
    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)
    input = input.permute(0, 2, 3, 1).contiguous().view(bs, height * width,
        n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(bs, height *
        width, n_instances)
    cluster_means = calculate_means(input, target, n_objects, max_n_objects,
        usegpu)
    var_term = calculate_variance_term(input, target, cluster_means,
        n_objects, delta_v, norm)
    dist_term = calculate_distance_term(cluster_means, n_objects, delta_d,
        norm, usegpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm)
    loss = alpha * var_term + beta * dist_term + gamma * reg_term
    return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm, size_average=True,
        reduce=True, usegpu=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce
        assert self.size_average
        assert self.reduce
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        _assert_no_grad(target)
        return discriminative_loss(input, target, n_objects, max_n_objects,
            self.delta_var, self.delta_dist, self.norm, self.usegpu)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Wizaron_instance_segmentation_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(CoordConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(CoordConvTranspose(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InstanceCounter(*[], **{'input_n_filters': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ReNet(*[], **{'n_input': 4, 'n_units': 4}), [torch.rand([4, 4, 4, 4])], {})

