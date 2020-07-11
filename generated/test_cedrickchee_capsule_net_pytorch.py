import sys
_module = sys.modules[__name__]
del sys
capsule_layer = _module
conv_layer = _module
decoder = _module
main = _module
model = _module
utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from torch.backends import cudnn


import torchvision.utils as vutils


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import datasets


class CapsuleLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing, num_routing, cuda_enabled):
        super(CapsuleLayer, self).__init__()
        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled
        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.

            This means PrimaryCapsules is composed of several convolutional units.
            """
            self.conv_units = nn.ModuleList([nn.Conv2d(self.in_channel, 32, 9, 2) for u in range(self.num_unit)])

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def routing(self, x):
        """
        Routing algorithm for capsule.

        :input: tensor x of shape [128, 8, 1152]

        :return: vector output of capsule j
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)
        u_hat = torch.matmul(batch_weight, x)
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            b_ij = b_ij
        num_iterations = self.num_routing
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = utils.squash(s_j, dim=3)
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.
        An example of a unit output shape is [128, 32, 6, 6]

        :return: vector output of capsule j
        """
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]
        unit = torch.stack(unit, dim=1)
        batch_size = x.size(0)
        unit = unit.view(batch_size, self.num_unit, -1)
        return utils.squash(unit, dim=2)


class ConvLayer(nn.Module):
    """
    Conventional Conv2d layer
    """

    def __init__(self, in_channel, out_channel, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass"""
        out_conv0 = self.conv0(x)
        out_relu = self.relu(out_conv0)
        return out_relu


class Decoder(nn.Module):
    """
    Implement Decoder structure in section 4.1, Figure 2 to reconstruct a digit
    from the `DigitCaps` layer representation.

    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.

    This Decoder network is used in training and prediction (testing).
    """

    def __init__(self, num_classes, output_unit_size, input_width, input_height, num_conv_in_channel, cuda_enabled):
        """
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 (or 3072 for CIFAR10) neurons each.
        """
        super(Decoder, self).__init__()
        self.cuda_enabled = cuda_enabled
        fc1_output_size = 512
        fc2_output_size = 1024
        self.fc3_output_size = input_width * input_height * num_conv_in_channel
        self.fc1 = nn.Linear(num_classes * output_unit_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size, self.fc3_output_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        """
        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the Decoder network, and
        reconstruct a [batch_size, fc3_output_size] size tensor representing the image.

        Args:
            x: [batch_size, 10, 16] The output of the digit capsule.
            target: [batch_size, 10] One-hot MNIST dataset labels.

        Returns:
            reconstruction: [batch_size, fc3_output_size] Tensor of reconstructed images.
        """
        batch_size = target.size(0)
        """
        First, do masking.
        """
        masked_caps = utils.mask(x, self.cuda_enabled)
        """
        Second, reconstruct the images with 3 Fully Connected layers.
        """
        vector_j = masked_caps.view(x.size(0), -1)
        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out))
        reconstruction = self.sigmoid(self.fc3(fc2_out))
        assert reconstruction.size() == torch.Size([batch_size, self.fc3_output_size])
        return reconstruction


class Net(nn.Module):
    """
    A simple CapsNet with 3 layers
    """

    def __init__(self, num_conv_in_channel, num_conv_out_channel, num_primary_unit, primary_unit_size, num_classes, output_unit_size, num_routing, use_reconstruction_loss, regularization_scale, input_width, input_height, cuda_enabled):
        """
        In the constructor we instantiate one ConvLayer module and two CapsuleLayer modules
        and assign them as member variables.
        """
        super(Net, self).__init__()
        self.cuda_enabled = cuda_enabled
        self.use_reconstruction_loss = use_reconstruction_loss
        self.image_width = input_width
        self.image_height = input_height
        self.image_channel = num_conv_in_channel
        self.regularization_scale = regularization_scale
        self.conv1 = ConvLayer(in_channel=num_conv_in_channel, out_channel=num_conv_out_channel, kernel_size=9)
        self.primary = CapsuleLayer(in_unit=0, in_channel=num_conv_out_channel, num_unit=num_primary_unit, unit_size=primary_unit_size, use_routing=False, num_routing=num_routing, cuda_enabled=cuda_enabled)
        self.digits = CapsuleLayer(in_unit=num_primary_unit, in_channel=primary_unit_size, num_unit=num_classes, unit_size=output_unit_size, use_routing=True, num_routing=num_routing, cuda_enabled=cuda_enabled)
        if use_reconstruction_loss:
            self.decoder = Decoder(num_classes, output_unit_size, input_width, input_height, num_conv_in_channel, cuda_enabled)

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        """
        out_conv1 = self.conv1(x)
        out_primary_caps = self.primary(out_conv1)
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, image, out_digit_caps, target, size_average=True):
        """Custom loss function

        Args:
            image: [batch_size, 1, 28, 28] MNIST samples.
            out_digit_caps: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
            target: [batch_size, 10] One-hot MNIST dataset labels.
            size_average: A boolean to enable mean loss (average loss over batch size).

        Returns:
            total_loss: A scalar Variable of total loss.
            m_loss: A scalar of margin loss.
            recon_loss: A scalar of reconstruction loss.
        """
        recon_loss = 0
        m_loss = self.margin_loss(out_digit_caps, target)
        if size_average:
            m_loss = m_loss.mean()
        total_loss = m_loss
        if self.use_reconstruction_loss:
            reconstruction = self.decoder(out_digit_caps, target)
            recon_loss = self.reconstruction_loss(reconstruction, image)
            if size_average:
                recon_loss = recon_loss.mean()
            total_loss = m_loss + recon_loss * self.regularization_scale
        return total_loss, m_loss, recon_loss * self.regularization_scale

    def margin_loss(self, input, target):
        """
        Class loss

        Implement equation 4 in section 3 'Margin loss for digit existence' in the paper.

        Args:
            input: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
            target: target: [batch_size, 10] One-hot MNIST labels.

        Returns:
            l_c: A scalar of class loss or also know as margin loss.
        """
        batch_size = input.size(0)
        v_c = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1) ** 2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1) ** 2
        t_c = target
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)
        return l_c

    def reconstruction_loss(self, reconstruction, image):
        """
        The reconstruction loss is the sum of squared differences between
        the reconstructed image (outputs of the logistic units) and
        the original image (input image).

        Implement section 4.1 'Reconstruction as a regularization method' in the paper.

        Based on naturomics's implementation.

        Args:
            reconstruction: [batch_size, 784] Decoder outputs of reconstructed image tensor.
            image: [batch_size, 1, 28, 28] MNIST samples.

        Returns:
            recon_error: A scalar Variable of reconstruction loss.
        """
        batch_size = image.size(0)
        image = image.view(batch_size, -1)
        error = reconstruction - image
        squared_error = error ** 2
        recon_error = torch.sum(squared_error, dim=1)
        return recon_error


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvLayer,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cedrickchee_capsule_net_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

