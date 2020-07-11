import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
main = _module
model = _module
modules = _module
plot_glimpses = _module
test_model = _module
test_modules = _module
test_retina = _module
trainer = _module
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


import numpy as np


import torch


from torchvision import datasets


from torchvision import transforms


from torch.utils.data.sampler import SubsetRandomSampler


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Normal


from functools import reduce


import matplotlib.patches as patches


import matplotlib.pyplot as plt


import time


from torch.optim.lr_scheduler import ReduceLROnPlateau


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()
        self.std = std
        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()
        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi
        return h_t, l_t, b_t, log_pi


class Retina:
    """A visual retina.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        g: size of the first square patch.
        k: number of patches to extract in the glimpse.
        s: scaling factor that controls the size of
            successive patches.

    Returns:
        phi: a 5D tensor of shape (B, k, g, g, C). The
            foveated glimpse of the image.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)
        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2).
        size: a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape
        start = self.denormalize(H, l)
        end = start + size
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))
        patch = []
        for i in range(B):
            patch.append(x[(i), :, start[i, 1]:end[i, 1], start[i, 0]:end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, T, coords):
        """Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if from_x < 0 or from_y < 0 or to_x > T or to_y > T:
            return True
        return False


class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()
        self.retina = Retina(g, k, s)
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev):
        phi = self.retina.foveate(x, l_t_prev)
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))
        what = self.fc3(phi_out)
        where = self.fc4(l_out)
        g_t = F.relu(what + where)
        return g_t


class CoreNetwork(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()
        self.std = std
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        l_t = torch.distributions.Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)
        l_t = torch.clamp(l_t, -1, 1)
        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActionNetwork,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaselineNetwork,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoreNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocationNetwork,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kevinzakka_recurrent_visual_attention(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

