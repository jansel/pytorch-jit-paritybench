import sys
_module = sys.modules[__name__]
del sys
CapsNet = _module
Decoder = _module
DigitCaps = _module
PrimaryCaps = _module
main = _module
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


import time


import torchvision.utils as vutils


from torchvision import datasets


from torchvision import transforms


class Decoder(nn.Module):
    """
    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.

    Reference: Section 4.1, Fig. 2
    """

    def __init__(self, opt):
        """
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 neurons each.
        """
        super(Decoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, v, target):
        """
        Args:
            `v`: [batch_size, 10, 16]
            `target`: [batch_size, 10]

        Return:
            `reconstruction`: [batch_size, 784]

        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the decoder network, and
        reconstruct a [batch_size, 784] size tensor representing the image.
        """
        batch_size = target.size(0)
        target = target.type(torch.FloatTensor)
        mask = torch.stack([target for i in range(16)], dim=2)
        assert mask.size() == torch.Size([batch_size, 10, 16])
        if self.opt.use_cuda & torch.cuda.is_available():
            mask = mask
        v_masked = mask * v
        v_masked = torch.sum(v_masked, dim=1)
        assert v_masked.size() == torch.Size([batch_size, 16])
        v = F.relu(self.fc1(v_masked))
        v = F.relu(self.fc2(v))
        reconstruction = torch.sigmoid(self.fc3(v))
        assert reconstruction.size() == torch.Size([batch_size, 784])
        return reconstruction


class DigitCaps(nn.Module):
    """
    The `DigitCaps` layer consists of 10 16D capsules. Compared to the traditional
    scalar output neurons in fully connected networks(FCN), the `DigitCaps` layer
    can be seen as an FCN with ten 16-dimensional output neurons, which we call
    these neurons "capsules".

    In this layer, we take the input `[1152, 8]` tensor `u` as 1152 [8,] vectors
    `u_i`, each `u_i` is a 8D output of the capsules from `PrimaryCaps` (see Eq.2
    in Section 2, Page 2) and sent to the 10 capsules. For each capsule, the tensor
    is first transformed by `W_ij`s into [1152, 16] size. Then we perform the Dynamic
    Routing algorithm to get the output `v_j` of size [16,]. As there are 10 capsules,
    the final output is [16, 10] size.

    #### Dimension transformation in this layer(ignoring `batch_size`):
    [1152, 8] --> [1152, 16] --> [1, 16] x 10 capsules --> [10, 16] output

    Note that in our codes we have vectorized these computations, so the dimensions
    above are just for understanding, actual dimensions of tensors are different.
    """

    def __init__(self, opt):
        """
        There is only one parameter in this layer, `W` [1, 1152, 10, 16, 8], where
        every [8, 16] is a weight matrix W_ij in Eq.2, that is, there are 11520
        `W_ij`s in total.

        The the coupling coefficients `b` [64, 1152, 10, 1] is a temporary variable which
        does NOT belong to the layer's parameters. In other words, `b` is not updated
        by gradient back-propagations. Instead, we update `b` by Dynamic Routing
        in every forward propagation. See the docstring of `self.forward` for details.
        """
        super(DigitCaps, self).__init__()
        self.opt = opt
        self.W = nn.Parameter(torch.randn(1, 1152, 10, 8, 16))

    def forward(self, u):
        """
        Args:
            `u`: [batch_size, 1152, 8]
        Return:
            `v`: [batch_size, 10, 16]

        In this layer, we vectorize our computations by calling `W` and using
        `torch.matmul()`. Thus the full computaion steps are as follows.
            1. Expand `W` into batches and compute `u_hat` (Eq.2)
            2. Line 2: Initialize `b` into zeros
            3. Line 3: Start Routing for `r` iterations:
                1. Line 4: c = softmax(b)
                2. Line 5: s = sum(c * u_hat)
                3. Line 6: v = squash(s)
                4. Line 7: b += u_hat * v

        The coupling coefficients `b` can be seen as a kind of attention matrix
        in the attentional sequence-to-sequence networks, which is widely used in
        Neural Machine Translation systems. For tutorials on  attentional seq2seq
        models, see https://arxiv.org/abs/1703.01619 or
        http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

        Reference: Section 2, Procedure 1
        """
        batch_size = u.size(0)
        assert u.size() == torch.Size([batch_size, 1152, 8])
        u = torch.unsqueeze(u, dim=2)
        u = torch.unsqueeze(u, dim=2)
        u_hat = torch.matmul(u, self.W).squeeze()
        b = Variable(torch.zeros(batch_size, 1152, 10, 1))
        if self.opt.use_cuda & torch.cuda.is_available():
            b = b
        for r in range(self.opt.r):
            c = F.softmax(b, dim=2)
            assert c.size() == torch.Size([batch_size, 1152, 10, 1])
            s = torch.sum(u_hat * c, dim=1)
            v = self.squash(s)
            assert v.size() == torch.Size([batch_size, 10, 16])
            a = u_hat * v.unsqueeze(1)
            b = b + torch.sum(a, dim=3, keepdim=True)
        return v

    def squash(self, s):
        """
        Args:
            `s`: [batch_size, 10, 16]

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        """
        batch_size = s.size(0)
        square = s ** 2
        square_sum = torch.sum(square, dim=2)
        norm = torch.sqrt(square_sum)
        factor = norm ** 2 / (norm * (1 + norm ** 2))
        v = factor.unsqueeze(2) * s
        assert v.size() == torch.Size([batch_size, 10, 16])
        return v


class PrimaryCaps(nn.Module):
    """
    The `PrimaryCaps` layer consists of 32 capsule units. Each unit takes
    the output of the `Conv1` layer, which is a `[256, 20, 20]` feature
    tensor (omitting `batch_size`), and performs a 2D convolution with 8
    output channels, kernel size 9 and stride 2, thus outputing a [8, 6, 6]
    tensor. In other words, you can see these 32 capsules as 32 paralleled 2D
    convolutional layers. Then we concatenate these 32 capsules' outputs and
    flatten them into a tensor of size `[1152, 8]`, representing 1152 8D
    vectors, and send it to the next layer `DigitCaps`.

    As indicated in Section 4, Page 4 in the paper, *One can see PrimaryCaps
    as a Convolution layer with Eq.1 as its block non-linearity.*, outputs of
    the `PrimaryCaps` layer are squashed before being passed to the next layer.

    Reference: Section 4, Fig. 1
    """

    def __init__(self):
        """
        We build 8 capsule units in the `PrimaryCaps` layer, each can be
        seen as a 2D convolution layer.
        """
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=256, out_channels=8, kernel_size=9, stride=2) for i in range(32)])

    def forward(self, x):
        """
        Each capsule outputs a [batch_size, 8, 6, 6] tensor, we need to
        flatten and concatenate them into a [batch_size, 8, 6*6, 32] size
        tensor and flatten and transpose into `u` [batch_size, 1152, 8], 
        where each [batch_size, 1152, 1] size tensor is the `u_i` in Eq.2. 

        #### Dimension transformation in this layer(ignoring `batch_size`):
        [256, 20, 20] --> [8, 6, 6] x 32 capsules --> [1152, 8]

        Note: `u_i` is one [1, 8] in the final [1152, 8] output, thus there are
        1152 `u_i`s.
        """
        batch_size = x.size(0)
        u = []
        for i in range(32):
            assert x.data.size() == torch.Size([batch_size, 256, 20, 20])
            u_i = self.capsules[i](x)
            assert u_i.size() == torch.Size([batch_size, 8, 6, 6])
            u_i = u_i.view(batch_size, 8, -1, 1)
            u.append(u_i)
        u = torch.cat(u, dim=3)
        u = u.view(batch_size, 8, -1)
        u = torch.transpose(u, 1, 2)
        assert u.data.size() == torch.Size([batch_size, 1152, 8])
        u_squashed = self.squash(u)
        return u_squashed

    def squash(self, u):
        """
        Args:
            `u`: [batch_size, 1152, 8]

        Return:
            `u_squashed`: [batch_size, 1152, 8]

        In CapsNet, we use the squash function after the output of both 
        capsule layers. Squash functions can be seen as activating functions
        like sigmoid, but for capsule layers rather than traditional fully
        connected layers, as they squash vectors instead of scalars.

        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))

        Reference: Eq.1 in Section 2.
        """
        batch_size = u.size(0)
        square = u ** 2
        square_sum = torch.sum(square, dim=2)
        norm = torch.sqrt(square_sum)
        factor = norm ** 2 / (norm * (1 + norm ** 2))
        u_squashed = factor.unsqueeze(2) * u
        assert u_squashed.size() == torch.Size([batch_size, 1152, 8])
        return u_squashed


class CapsNet(nn.Module):

    def __init__(self, opt):
        """
        The CapsNet consists of 3 layers: `Conv1`, `PrimaryCaps`, `DigitCaps`.`Conv1`
        is an ordinary 2D convolutional layer with 9x9 kernels, stride 2, 256 output
        channels, and ReLU activations. `PrimaryCaps` and `DigitCaps` are two capsule
        layers with Dynamic Routing between them. For further details of these two
        layers, see the docstrings of their classes. For each [1, 28, 28] input image,
        CapsNet outputs a [16, 10] tensor, representing the 16-dimensional output
        vector from 10 digit capsules.

        Reference: Section 4, Figure 1
        """
        super(CapsNet, self).__init__()
        self.opt = opt
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9)
        self.PrimaryCaps = PrimaryCaps()
        self.DigitCaps = DigitCaps(opt)
        self.Decoder = Decoder(opt)

    def forward(self, x):
        """
        Args:
            `x`: [batch_size, 1, 28, 28] MNIST samples
        
        Return:
            `v`: [batch_size, 10, 16] CapsNet outputs, 16D prediction vectors of
                10 digit capsules

        The dimension transformation procedure of an input tensor in each layer:
            0. Input: [batch_size, 1, 28, 28] -->
            1. `Conv1` --> [batch_size, 256, 20, 20] --> 
            2. `PrimaryCaps` --> [batch_size, 8, 6, 6] x 32 capsules --> 
            3. Flatten, concatenate, squash --> [batch_size, 1152, 8] -->
            4. `W_ij`s and `DigitCaps` --> [batch_size, 16, 10] -->
            5. Length of 10 capsules --> [batch_size, 10] output probabilities
        """
        x = F.relu(self.Conv1(x))
        u = self.PrimaryCaps(x)
        v = self.DigitCaps(u)
        return v

    def marginal_loss(self, v, target, l=0.5):
        """
        Args:
            `v`: [batch_size, 10, 16]
            `target`: [batch_size, 10]
            `l`: Scalar, lambda for down-weighing the loss for absent digit classes

        Return:
            `marginal_loss`: Scalar
        
        L_c = T_c * max(0, m_plus - norm(v_c)) ^ 2 + lambda * (1 - T_c) * max(0, norm(v_c) - m_minus) ^2
        
        Reference: Eq.4 in Section 3.
        """
        batch_size = v.size(0)
        square = v ** 2
        square_sum = torch.sum(square, dim=2)
        norm = torch.sqrt(square_sum)
        assert norm.size() == torch.Size([batch_size, 10])
        T_c = target.type(torch.FloatTensor)
        zeros = Variable(torch.zeros(norm.size()))
        if self.opt.use_cuda & torch.cuda.is_available():
            zeros = zeros
            T_c = T_c
        marginal_loss = T_c * torch.max(zeros, 0.9 - norm) ** 2 + (1 - T_c) * l * torch.max(zeros, norm - 0.1) ** 2
        marginal_loss = torch.sum(marginal_loss)
        return marginal_loss

    def reconstruction_loss(self, reconstruction, image):
        """
        Args:
            `reconstruction`: [batch_size, 784] Decoder outputs of images
            `image`: [batch_size, 1, 28, 28] MNIST samples

        Return:
            `reconstruction_loss`: Scalar Variable

        The reconstruction loss is measured by a squared differences
        between the reconstruction and the original image. 

        Reference: Section 4.1
        """
        batch_size = image.size(0)
        image = image.view(batch_size, -1)
        assert image.size() == (batch_size, 784)
        reconstruction_loss = torch.sum((reconstruction - image) ** 2)
        return reconstruction_loss

    def loss(self, v, target, image):
        """
        Args:
            `v`: [batch_size, 10, 16] CapsNet outputs
            `target`: [batch_size, 10] One-hot MNIST labels
            `image`: [batch_size, 1, 28, 28] MNIST samples

        Return:
            `L`: Scalar Variable, total loss
            `marginal_loss`: Scalar Variable
            `reconstruction_loss`: Scalar Variable

        The reconstruction loss is scaled down by 5e-4, serving as a
        regularization method.

        Reference: Section 4.1
        """
        batch_size = image.size(0)
        marginal_loss = self.marginal_loss(v, target)
        reconstruction = self.Decoder(v, target)
        reconstruction_loss = self.reconstruction_loss(reconstruction, image)
        loss = (marginal_loss + 0.0005 * reconstruction_loss) / batch_size
        return loss, marginal_loss / batch_size, reconstruction_loss / batch_size

