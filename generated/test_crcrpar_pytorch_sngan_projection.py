import sys
_module = sys.modules[__name__]
del sys
evaluation = _module
links = _module
conditional_batchnorm = _module
losses = _module
metrics = _module
fid = _module
models = _module
discriminators = _module
resblocks = _module
snresnet = _module
snresnet64 = _module
generators = _module
resblocks = _module
resnet = _module
resnet64 = _module
inception = _module
train_64 = _module
utils = _module

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


import torch.nn.functional as F


from torch.nn import init


import torch


import numpy as np


from scipy import linalg


import math


from torch.nn import utils


import torch.utils.data as data


import torch.optim as optim


import numpy


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False,
        track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(num_features, eps,
            momentum, affine, track_running_stats)

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = (1.0 / self.
                    num_batches_tracked.item())
            else:
                exponential_average_factor = self.momentum
        output = F.batch_norm(input, self.running_mean, self.running_var,
            self.weight, self.bias, self.training or not self.
            track_running_stats, exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, activation
        =F.relu, downsample=False):
        super(Block, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_ch != out_ch or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch
        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2, activation=
            activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4, activation=
            activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8, activation=
            activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16, activation
            =activation, downsample=True)
        self.block6 = Block(num_features * 16, num_features * 16,
            activation=activation, downsample=True)
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, 
                num_features * 16))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes, activation=F.relu,
        dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation
        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2, activation=
            activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4, activation=
            activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16, activation
            =activation, downsample=True)
        self.block6 = Block(num_features * 16, num_features * 16,
            activation=activation, downsample=False)
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 4):
            h = getattr(self, 'block{}'.format(i))(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        for i in range(4, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = torch.sum(self.activation(h), dim=(2, 3))
        return self.l7(h)


class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2, activation=
            activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4, activation=
            activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8, activation=
            activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16, activation
            =activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, 
                num_features * 16))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes, activation=F.relu,
        dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation
        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2, activation=
            activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4, activation=
            activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16, activation
            =activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        h = self.block4(h)
        h = self.block5(h)
        h = torch.sum(self.activation(h), dim=(2, 3))
        return self.l6(h)


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-05, momentum=0.1,
        affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(num_features,
            eps, momentum, affine, track_running_stats)
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)
        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)
        return super(CategoricalConditionalBatchNorm2d, self).forward(input,
            weight, bias)


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, activation
        =F.relu, upsample=False, num_classes=0):
        super(Block, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class ResNetGenerator(nn.Module):
    """Generator generates 128x128."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
        activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution
        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width ** 2)
        self.block2 = Block(num_features * 16, num_features * 16,
            activation=activation, upsample=True, num_classes=num_classes)
        self.block3 = Block(num_features * 16, num_features * 8, activation
            =activation, upsample=True, num_classes=num_classes)
        self.block4 = Block(num_features * 8, num_features * 4, activation=
            activation, upsample=True, num_classes=num_classes)
        self.block5 = Block(num_features * 4, num_features * 2, activation=
            activation, upsample=True, num_classes=num_classes)
        self.block6 = Block(num_features * 2, num_features, activation=
            activation, upsample=True, num_classes=num_classes)
        self.b7 = nn.BatchNorm2d(num_features)
        self.conv7 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width
            )
        for i in [2, 3, 4, 5, 6]:
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return torch.tanh(self.conv7(h))


class ResNetGenerator(nn.Module):
    """Generator generates 64x64."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
        activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution
        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width ** 2)
        self.block2 = Block(num_features * 16, num_features * 8, activation
            =activation, upsample=True, num_classes=num_classes)
        self.block3 = Block(num_features * 8, num_features * 4, activation=
            activation, upsample=True, num_classes=num_classes)
        self.block4 = Block(num_features * 4, num_features * 2, activation=
            activation, upsample=True, num_classes=num_classes)
        self.block5 = Block(num_features * 2, num_features, activation=
            activation, upsample=True, num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width
            )
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=
        True, normalize_input=True, requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.
                MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.
                Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception
                .Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.
                Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, (0)] = x[:, (0)] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, (1)] = x[:, (1)] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, (2)] = x[:, (2)] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_crcrpar_pytorch_sngan_projection(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Block(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(OptimizedBlock(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

