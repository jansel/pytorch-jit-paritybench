import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
torchgan = _module
test_layers = _module
test_losses = _module
test_metrics = _module
test_models = _module
test_trainer = _module
layers = _module
denseblock = _module
minibatchdiscrimination = _module
residual = _module
selfattention = _module
spectralnorm = _module
virtualbatchnorm = _module
logging = _module
backends = _module
logger = _module
visualize = _module
losses = _module
auxclassifier = _module
boundaryequilibrium = _module
draganpenalty = _module
energybased = _module
featurematching = _module
functional = _module
historical = _module
leastsquares = _module
loss = _module
minimax = _module
mutualinfo = _module
wasserstein = _module
metrics = _module
classifierscore = _module
metric = _module
models = _module
acgan = _module
autoencoding = _module
conditional = _module
dcgan = _module
infogan = _module
model = _module
trainer = _module
base_trainer = _module
parallel_trainer = _module
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


import torch


import torch.distributions as distributions


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Parameter


import torch.autograd as autograd


from math import ceil


from math import log


from math import log2


class BasicBlock2d(nn.Module):
    """Basic Block Module as described in `"Densely Connected Convolutional Networks by Huang et.
    al." <https://arxiv.org/abs/1608.06993>`_

    The output is computed by ``concatenating`` the ``input`` tensor to the ``output`` tensor (of the
    internal model) along the ``channel`` dimension.

    The internal model is simply a sequence of a ``Conv2d`` layer and a ``BatchNorm2d`` layer, if
    activated.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding
        =0, batchnorm=True, nonlinearity=None):
        super(BasicBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nl, nn.
                Conv2d(in_channels, out_channels, kernel, stride, padding,
                bias=False))
        else:
            self.model = nn.Sequential(nl, nn.Conv2d(in_channels,
                out_channels, kernel, stride, padding, bias=True))

    def forward(self, x):
        """Computes the output of the basic dense block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by concatenating the input to the output of the internal model.
        """
        return torch.cat([x, self.model(x)], 1)


class BottleneckBlock2d(nn.Module):
    """Bottleneck Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    The output is computed by ``concatenating`` the ``input`` tensor to the ``output`` tensor (of the
    internal model) along the ``channel`` dimension.

    The internal model is simply a sequence of 2 ``Conv2d`` layers and 2 ``BatchNorm2d`` layers, if
    activated. This Module is much more computationally efficient than the ``BasicBlock2d``, and hence
    is more recommended.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        bottleneck_channels (int, optional): The channels in the intermediate convolutional
                                             layer. A higher value will make learning of
                                             more complex functions possible. Defaults to
                                             ``4 * in_channels``.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding
        =0, bottleneck_channels=None, batchnorm=True, nonlinearity=None):
        super(BottleneckBlock2d, self).__init__()
        bottleneck_channels = (4 * in_channels if bottleneck_channels is
            None else bottleneck_channels)
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nl, nn.
                Conv2d(in_channels, bottleneck_channels, 1, 1, 0, bias=
                False), nn.BatchNorm2d(bottleneck_channels), nl, nn.Conv2d(
                bottleneck_channels, out_channels, kernel, stride, padding,
                bias=False))
        else:
            self.model = nn.Sequential(nl, nn.Conv2d(in_channels,
                bottleneck_channels, 1, 1, 0, bias=True), nl, nn.Conv2d(
                bottleneck_channels, out_channels, kernel, stride, padding,
                bias=True))

    def forward(self, x):
        """Computes the output of the bottleneck dense block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by concatenating the input to the output of the internal model.
        """
        return torch.cat([x, self.model(x)], 1)


class TransitionBlock2d(nn.Module):
    """Transition Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    This is a simple ``Sequential`` model of a ``Conv2d`` layer and a ``BatchNorm2d`` layer, if
    activated.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding
        =0, batchnorm=True, nonlinearity=None):
        super(TransitionBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nl, nn.
                Conv2d(in_channels, out_channels, kernel, stride, padding,
                bias=False))
        else:
            self.model = nn.Sequential(nl, nn.Conv2d(in_channels,
                out_channels, kernel, stride, padding, bias=True))

    def forward(self, x):
        """Computes the output of the transition block

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)


class TransitionBlockTranspose2d(nn.Module):
    """Transition Block Transpose Module is constructed by simply reversing the effect of
    Transition Block Module. We replace the ``Conv2d`` layers by ``ConvTranspose2d`` layers.

    Args:
        in_channels (int): The channel dimension of the input tensor.
        out_channels (int): The channel dimension of the output tensor.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding
        =0, batchnorm=True, nonlinearity=None):
        super(TransitionBlockTranspose2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if batchnorm is True:
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nl, nn.
                ConvTranspose2d(in_channels, out_channels, kernel, stride,
                padding, bias=False))
        else:
            self.model = nn.Sequential(nl, nn.ConvTranspose2d(in_channels,
                out_channels, kernel, stride, padding, bias=True))

    def forward(self, x):
        """Computes the output of the transition block transpose

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)


class DenseBlock2d(nn.Module):
    """Dense Block Module as described in `"Densely Connected Convolutional Networks by Huang
    et. al." <https://arxiv.org/abs/1608.06993>`_

    Args:
        depth (int): The total number of ``blocks`` that will be present.
        in_channels (int): The channel dimension of the input tensor.
        growth_rate (int): The rate at which the channel dimension increases. The output of
                           the module has a channel dimension of size ``in_channels +
                           depth * growth_rate``.
        block (torch.nn.Module): Should be once of the Densenet Blocks. Forms the building block
                                 for the Dense Block.
        kernel (int, tuple): Size of the Convolutional Kernel.
        stride (int, tuple, optional): Stride of the Convolutional Kernel.
        padding (int, tuple, optional): Padding to be applied on the input tensor.
        batchnorm (bool, optional): If ``True``, batch normalization shall be performed.
        nonlinearity (torch.nn.Module, optional): Activation to be applied. Defaults to
                                                  ``torch.nn.LeakyReLU``.
    """

    def __init__(self, depth, in_channels, growth_rate, block, kernel,
        stride=1, padding=0, batchnorm=True, nonlinearity=None):
        super(DenseBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        model = []
        for i in range(depth):
            model.append(block(in_channels + i * growth_rate, growth_rate,
                kernel, stride, padding, batchnorm=batchnorm, nonlinearity=nl))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Computes the output of the transition block transpose

        Args:
            x (torch.Tensor): The input tensor having channel dimension same as ``in_channels``.

        Returns:
            4D Tensor by applying the ``model`` on ``x``.
        """
        return self.model(x)


class MinibatchDiscrimination1d(nn.Module):
    """1D Minibatch Discrimination Module as proposed in the paper `"Improved Techniques for
    Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Allows the Discriminator to easily detect mode collapse by augmenting the activations to the succeeding
    layer with side information that allows it to determine the 'closeness' of the minibatch examples
    with each other

    .. math :: M_i = T * f(x_{i})
    .. math :: c_b(x_{i}, x_{j}) = \\exp(-||M_{i, b} - M_{j, b}||_1) \\in \\mathbb{R}.
    .. math :: o(x_{i})_b &= \\sum_{j=1}^{n} c_b(x_{i},x_{j}) \\in \\mathbb{R} \\\\
    .. math :: o(x_{i}) &= \\Big[ o(x_{i})_1, o(x_{i})_2, \\dots, o(x_{i})_B \\Big] \\in \\mathbb{R}^B \\\\
    .. math :: o(X) \\in \\mathbb{R}^{n \\times B}

    This is followed by concatenating :math:`o(x_{i})` and :math:`f(x_{i})`

    where

    - :math:`f(x_{i}) \\in \\mathbb{R}^A` : Activations from an intermediate layer
    - :math:`f(x_{i}) \\in \\mathbb{R}^A` : Parameter Tensor for generating minibatch discrimination matrix


    Args:
        in_features (int): Features input corresponding to dimension :math:`A`
        out_features (int): Number of output features that are to be concatenated corresponding to dimension :math:`B`
        intermediate_features (int): Intermediate number of features corresponding to dimension :math:`C`

    Returns:
        A Tensor of size :math:`(N, in_features + out_features)` where :math:`N` is the batch size
    """

    def __init__(self, in_features, out_features, intermediate_features=16):
        super(MinibatchDiscrimination1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features
        self.T = nn.Parameter(torch.Tensor(in_features, out_features,
            intermediate_features))
        nn.init.normal_(self.T)

    def forward(self, x):
        """Computes the output of the Minibatch Discrimination Layer

        Args:
            x (torch.Tensor): A Torch Tensor of dimensions :math: `(N, infeatures)`

        Returns:
            3D Torch Tensor of size :math: `(N,infeatures + outfeatures)` after applying Minibatch Discrimination
        """
        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features, self.intermediate_features
            ).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        out = torch.sum(torch.exp(-torch.abs(M - M_t).sum(3)), dim=0) - 1
        return torch.cat([x, out], 1)


class ResidualBlock2d(nn.Module):
    """Residual Block Module as described in `"Deep Residual Learning for Image Recognition
    by He et. al." <https://arxiv.org/abs/1512.03385>`_

    The output of the residual block is computed in the following manner:

    .. math:: output = activation(layers(x) + shortcut(x))

    where

    - :math:`x` : Input to the Module
    - :math:`layers` : The feed forward network
    - :math:`shortcut` : The function to be applied along the skip connection
    - :math:`activation` : The activation function applied at the end of the residual block

    Args:
        filters (list): A list of the filter sizes. For ex, if the input has a channel
            dimension of 16, and you want 3 convolution layers and the final output to have a
            channel dimension of 16, then the list would be [16, 32, 64, 16].
        kernels (list): A list of the kernel sizes. Each kernel size can be an integer or a
            tuple, similar to Pytorch convention. The length of the ``kernels`` list must be
            1 less than the ``filters`` list.
        strides (list, optional): A list of the strides for each convolution layer.
        paddings (list, optional): A list of the padding in each convolution layer.
        nonlinearity (torch.nn.Module, optional): The activation to be used after every convolution
            layer.
        batchnorm (bool, optional): If set to ``False``, batch normalization is not used after
            every convolution layer.
        shortcut (torch.nn.Module, optional): The function to be applied on the input along the
            skip connection.
        last_nonlinearity (torch.nn.Module, optional): The activation to be applied at the end of
            the residual block.
    """

    def __init__(self, filters, kernels, strides=None, paddings=None,
        nonlinearity=None, batchnorm=True, shortcut=None, last_nonlinearity
        =None):
        super(ResidualBlock2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if strides is None:
            strides = [(1) for _ in range(len(kernels))]
        if paddings is None:
            paddings = [(0) for _ in range(len(kernels))]
        assert len(filters) == len(kernels) + 1 and len(filters) == len(strides
            ) + 1 and len(filters) == len(paddings) + 1
        layers = []
        for i in range(1, len(filters)):
            layers.append(nn.Conv2d(filters[i - 1], filters[i], kernels[i -
                1], strides[i - 1], paddings[i - 1]))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            if i != len(filters):
                layers.append(nl)
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut
        self.last_nonlinearity = last_nonlinearity

    def forward(self, x):
        """Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        return (out if self.last_nonlinearity is None else self.
            last_nonlinearity(out))


class ResidualBlockTranspose2d(nn.Module):
    """A customized version of Residual Block having Conv Transpose layers instead of Conv layers.

    The output of this block is computed in the following manner:

    .. math:: output = activation(layers(x) + shortcut(x))

    where

    - :math:`x` : Input to the Module
    - :math:`layers` : The feed forward network
    - :math:`shortcut` : The function to be applied along the skip connection
    - :math:`activation` : The activation function applied at the end of the residual block

    Args:
        filters (list): A list of the filter sizes. For ex, if the input has a channel
            dimension of 16, and you want 3 transposed convolution layers and the final output
            to have a channel dimension of 16, then the list would be [16, 32, 64, 16].
        kernels (list): A list of the kernel sizes. Each kernel size can be an integer or a
            tuple, similar to Pytorch convention. The length of the ``kernels`` list must be
            1 less than the ``filters`` list.
        strides (list, optional): A list of the strides for each convolution layer.
        paddings (list, optional): A list of the padding in each convolution layer.
        nonlinearity (torch.nn.Module, optional): The activation to be used after every convolution
            layer.
        batchnorm (bool, optional): If set to ``False``, batch normalization is not used after
            every convolution layer.
        shortcut (torch.nn.Module, optional): The function to be applied on the input along the
            skip connection.
        last_nonlinearity (torch.nn.Module, optional): The activation to be applied at the end of
            the residual block.
    """

    def __init__(self, filters, kernels, strides=None, paddings=None,
        nonlinearity=None, batchnorm=True, shortcut=None, last_nonlinearity
        =None):
        super(ResidualBlockTranspose2d, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        if strides is None:
            strides = [(1) for _ in range(len(kernels))]
        if paddings is None:
            paddings = [(0) for _ in range(len(kernels))]
        assert len(filters) == len(kernels) + 1 and len(filters) == len(strides
            ) + 1 and len(filters) == len(paddings) + 1
        layers = []
        for i in range(1, len(filters)):
            layers.append(nn.ConvTranspose2d(filters[i - 1], filters[i],
                kernels[i - 1], strides[i - 1], paddings[i - 1]))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            if i != len(filters):
                layers.append(nl)
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut
        self.last_nonlinearity = last_nonlinearity

    def forward(self, x):
        """Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Transposed Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        return (out if self.last_nonlinearity is None else self.
            last_nonlinearity(out))


class SelfAttention2d(nn.Module):
    """Self Attention Module as proposed in the paper `"Self-Attention Generative Adversarial
    Networks by Han Zhang et. al." <https://arxiv.org/abs/1805.08318>`_

    .. math:: attention = softmax((query(x))^T * key(x))
    .. math:: output = \\gamma * value(x) * attention + x

    where

    - :math:`query` : 2D Convolution Operation
    - :math:`key` : 2D Convolution Operation
    - :math:`value` : 2D Convolution Operation
    - :math:`x` : Input

    Args:
        input_dims (int): The input channel dimension in the input ``x``.
        output_dims (int, optional): The output channel dimension. If ``None`` the output
            channel value is computed as ``input_dims // 8``. So if the ``input_dims`` is **less
            than 8** then the layer will give an error.
        return_attn (bool, optional): Set it to ``True`` if you want the attention values to be
            returned.
    """

    def __init__(self, input_dims, output_dims=None, return_attn=False):
        output_dims = input_dims // 8 if output_dims is None else output_dims
        if output_dims == 0:
            raise Exception(
                'The output dims corresponding to the input dims is 0. Increase the input                            dims to 8 or more. Else specify output_dims'
                )
        super(SelfAttention2d, self).__init__()
        self.query = nn.Conv2d(input_dims, output_dims, 1)
        self.key = nn.Conv2d(input_dims, output_dims, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attn = return_attn

    def forward(self, x):
        """Computes the output of the Self Attention Layer

        Args:
            x (torch.Tensor): A 4D Tensor with the channel dimension same as ``input_dims``.

        Returns:
            A tuple of the ``output`` and the ``attention`` if ``return_attn`` is set to ``True``
            else just the ``output`` tensor.
        """
        dims = x.size(0), -1, x.size(2) * x.size(3)
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        if self.return_attn:
            return out, attn
        return out


class SpectralNorm2d(nn.Module):
    """2D Spectral Norm Module as described in `"Spectral Normalization
    for Generative Adversarial Networks by Miyato et. al." <https://arxiv.org/abs/1802.05957>`_
    The spectral norm is computed using ``power iterations``.

    Computation Steps:

    .. math:: v_{t + 1} = \\frac{W^T W v_t}{||W^T W v_t||} = \\frac{(W^T W)^t v}{||(W^T W)^t v||}
    .. math:: u_{t + 1} = W v_t
    .. math:: v_{t + 1} = W^T u_{t + 1}
    .. math:: Norm(W) = ||W v|| = u^T W v
    .. math:: Output = \\frac{W}{Norm(W)} = \\frac{W}{u^T W v}

    Args:
        module (torch.nn.Module): The Module on which the Spectral Normalization needs to be
            applied.
        name (str, optional): The attribute of the ``module`` on which normalization needs to
            be performed.
        power_iterations (int, optional): Total number of iterations for the norm to converge.
            ``1`` is usually enough given the weights vary quite gradually.

    Example:
        .. code:: python

            >>> layer = SpectralNorm2d(Conv2d(3, 16, 1))
            >>> x = torch.rand(1, 3, 10, 10)
            >>> layer(x)
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm2d, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        self.u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=
            False)
        self.v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False
            )
        self.u.data = self._l2normalize(self.u.data)
        self.v.data = self._l2normalize(self.v.data)
        self.w_bar = Parameter(w.data)
        del self.module._parameters[self.name]

    def _l2normalize(self, x, eps=1e-12):
        """Function to calculate the ``L2 Normalized`` form of a Tensor

        Args:
            x (torch.Tensor): Tensor which needs to be normalized.
            eps (float, optional): A small value needed to avoid infinite values.

        Returns:
            Normalized form of the tensor ``x``.
        """
        return x / (torch.norm(x) + eps)

    def forward(self, *args):
        """Computes the output of the ``module`` and appies spectral normalization to the
        ``name`` attribute of the ``module``.

        Returns:
            The output of the ``module``.
        """
        height = self.w_bar.data.shape[0]
        for _ in range(self.power_iterations):
            self.v.data = self._l2normalize(torch.mv(torch.t(self.w_bar.
                view(height, -1)), self.u))
            self.u.data = self._l2normalize(torch.mv(self.w_bar.view(height,
                -1), self.v))
        sigma = self.u.dot(self.w_bar.view(height, -1).mv(self.v))
        setattr(self.module, self.name, self.w_bar / sigma.expand_as(self.
            w_bar))
        return self.module.forward(*args)


class VirtualBatchNorm(nn.Module):
    """Virtual Batch Normalization Module as proposed in the paper
    `"Improved Techniques for Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Performs Normalizes the features of a batch based on the statistics collected on a reference
    batch of samples that are chosen once and fixed from the start, as opposed to regular
    batch normalization that uses the statistics of the batch being normalized

    Virtual Batch Normalization requires that the size of the batch being normalized is at least
    a multiple of (and ideally equal to) the size of the reference batch. Keep this in mind while
    choosing the batch size in ```torch.utils.data.DataLoader``` or use ```drop_last=True```

    .. math:: y = \\frac{x - \\mathrm{E}[x_{ref}]}{\\sqrt{\\mathrm{Var}[x_{ref}] + \\epsilon}} * \\gamma + \\beta

    where

    - :math:`x` : Batch Being Normalized
    - :math:`x_{ref}` : Reference Batch

    Args:
        in_features (int): Size of the input dimension to be normalized
        eps (float, optional): Value to be added to variance for numerical stability while normalizing
    """

    def __init__(self, in_features, eps=1e-05):
        super(VirtualBatchNorm, self).__init__()
        self.in_features = in_features
        self.scale = nn.Parameter(torch.ones(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.ref_mu = None
        self.ref_var = None
        self.eps = eps

    def _batch_stats(self, x):
        """Computes the statistics of the batch ``x``.

        Args:
            x (torch.Tensor): Tensor whose statistics need to be computed.

        Returns:
            A tuple of the mean and variance of the batch ``x``.
        """
        mu = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        return mu, var

    def _normalize(self, x, mu, var):
        """Normalizes the tensor ``x`` using the statistics ``mu`` and ``var``.

        Args:
            x (torch.Tensor): The Tensor to be normalized.
            mu (torch.Tensor): Mean using which the Tensor is to be normalized.
            var (torch.Tensor): Variance used in the normalization of ``x``.

        Returns:
            Normalized Tensor ``x``.
        """
        std = torch.sqrt(self.eps + var)
        x = (x - mu) / std
        sizes = list(x.size())
        for dim, i in enumerate(x.size()):
            if dim != 1:
                sizes[dim] = 1
        scale = self.scale.view(*sizes)
        bias = self.bias.view(*sizes)
        return x * scale + bias

    def forward(self, x):
        """Computes the output of the Virtual Batch Normalization

        Args:
            x (torch.Tensor): A Torch Tensor of dimension at least 2 which is to be Normalized

        Returns:
            Torch Tensor of the same dimension after normalizing with respect to the statistics of the reference batch
        """
        assert x.size(1) == self.in_features
        if self.ref_mu is None or self.ref_var is None:
            self.ref_mu, self.ref_var = self._batch_stats(x)
            self.ref_mu = self.ref_mu.clone().detach()
            self.ref_var = self.ref_var.clone().detach()
            out = self._normalize(x, self.ref_mu, self.ref_var)
        else:
            out = self._normalize(x, self.ref_mu, self.ref_var)
            self.ref_mu = None
            self.ref_var = None
        return out


class GeneratorLoss(nn.Module):
    """Base class for all generator losses.

    .. note:: All Losses meant to be minimized for optimizing the Generator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction='mean', override_train_ops=None):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}

    def set_arg_map(self, value):
        """Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_generator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_generator,
        device, batch_size, labels=None):
        """Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value = discriminator(fake)`
        3. :math:`loss = loss\\_function(value)`
        4. Backpropagate by computing :math:`\\nabla loss`
        5. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator,
                optimizer_generator, device, batch_size, labels)
        else:
            if labels is None and generator.label_type == 'required':
                raise Exception('GAN model requires labels for training')
            noise = torch.randn(batch_size, generator.encoding_dims, device
                =device)
            optimizer_generator.zero_grad()
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (
                    batch_size,), device=device)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            elif generator.label_type == 'generated':
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake)
            elif generator.label_type == 'generated':
                dgz = discriminator(fake, label_gen)
            else:
                dgz = discriminator(fake, labels)
            loss = self.forward(dgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()


class DiscriminatorLoss(nn.Module):
    """Base class for all discriminator losses.

    .. note:: All Losses meant to be minimized for optimizing the Discriminator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction='mean', override_train_ops=None):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}

    def set_arg_map(self, value):
        """Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_discriminator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_discriminator,
        real_inputs, device, labels=None):
        """Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator,
                optimizer_discriminator, real_inputs, device, labels)
        else:
            if labels is None and (generator.label_type == 'required' or 
                discriminator.label_type == 'required'):
                raise Exception('GAN model requires labels for training')
            batch_size = real_inputs.size(0)
            noise = torch.randn(batch_size, generator.encoding_dims, device
                =device)
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (
                    batch_size,), device=device)
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == 'none':
                dx = discriminator(real_inputs)
            elif discriminator.label_type == 'required':
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake.detach())
            elif generator.label_type == 'generated':
                dgz = discriminator(fake.detach(), label_gen)
            else:
                dgz = discriminator(fake.detach(), labels)
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            return loss.item()


class Generator(nn.Module):
    """Base class for all Generator models. All Generator models must subclass this.

    Args:
        encoding_dims (int): Dimensions of the sample from the noise prior.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(self, encoding_dims, label_type='none'):
        super(Generator, self).__init__()
        self.encoding_dims = encoding_dims
        self.label_type = label_type

    def _weight_initializer(self):
        """Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def sampler(self, sample_size, device):
        """Function to allow sampling data at inference time. Models requiring
        input in any other format must override it in the subclass.

        Args:
            sample_size (int): The number of images to be generated
            device (torch.device): The device on which the data must be generated

        Returns:
            A list of the items required as input
        """
        return [torch.randn(sample_size, self.encoding_dims, device=device)]


class Discriminator(nn.Module):
    """Base class for all Discriminator models. All Discriminator models must subclass this.

    Args:
        input_dims (int): Dimensions of the input.
        label_type (str, optional): The type of labels expected by the Discriminator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(self, input_dims, label_type='none'):
        super(Discriminator, self).__init__()
        self.input_dims = input_dims
        self.label_type = label_type

    def _weight_initializer(self):
        """Default weight initializer for all disciminator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_torchgan_torchgan(_paritybench_base):
    pass
    def test_000(self):
        self._check(TransitionBlock2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(TransitionBlockTranspose2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(MinibatchDiscrimination1d(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(VirtualBatchNorm(*[], **{'in_features': 4}), [torch.rand([4, 4, 4, 4])], {})

