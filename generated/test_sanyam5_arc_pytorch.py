import sys
_module = sys.modules[__name__]
del sys
batcher = _module
download_data = _module
image_augmenter = _module
models = _module
test_models = _module
train = _module
viz = _module

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


from numpy.random import choice


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import math


use_cuda = False


class GlimpseWindow:
    """
    Generates glimpses from images using Cauchy kernels.

    Args:
        glimpse_h (int): The height of the glimpses to be generated.
        glimpse_w (int): The width of the glimpses to be generated.

    """

    def __init__(self, glimpse_h: int, glimpse_w: int):
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w

    @staticmethod
    def _get_filterbanks(delta_caps: Variable, center_caps: Variable, image_size: int, glimpse_size: int) ->Variable:
        """
        Generates Cauchy Filter Banks along a dimension.

        Args:
            delta_caps (B,):  A batch of deltas [-1, 1]
            center_caps (B,): A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            image_size (int): size of images along that dimension
            glimpse_size (int): size of glimpses to be generated along that dimension

        Returns:
            (B, image_size, glimpse_size): A batch of filter banks

        """
        image_size = float(image_size)
        glimpse_size = float(glimpse_size)
        centers = (image_size - 1) * (center_caps + 1) / 2.0
        deltas = float(image_size) / glimpse_size * (1.0 - torch.abs(delta_caps))
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))
        glimpse_pixels = Variable(torch.arange(0, glimpse_size) - (glimpse_size - 1.0) / 2.0)
        if use_cuda:
            glimpse_pixels = glimpse_pixels
        glimpse_pixels = deltas[:, (None)] * glimpse_pixels[(None), :]
        glimpse_pixels = centers[:, (None)] + glimpse_pixels
        image_pixels = Variable(torch.arange(0, image_size))
        if use_cuda:
            image_pixels = image_pixels
        fx = image_pixels - glimpse_pixels[:, :, (None)]
        fx = fx / gammas[:, (None), (None)]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, (None), (None)] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 0.0001)[:, :, (None)]
        return fx.transpose(1, 2)

    def get_attention_mask(self, glimpse_params: Variable, mask_h: int, mask_w: int) ->Variable:
        """
        For visualization, generate a heat map (or mask) of which pixels got the most "attention".

        Args:
            glimpse_params (B, hx):  A batch of glimpse parameters.
            mask_h (int): The height of the image for which the mask is being generated.
            mask_w (int): The width of the image for which the mask is being generated.

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended pixels weighted more.

        """
        batch_size, _ = glimpse_params.size()
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, (2)], center_caps=glimpse_params[:, (0)], image_size=mask_h, glimpse_size=self.glimpse_h)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, (2)], center_caps=glimpse_params[:, (1)], image_size=mask_w, glimpse_size=self.glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()
        return mask

    def get_glimpse(self, images: Variable, glimpse_params: Variable) ->Variable:
        """
        Generate glimpses given images and glimpse parameters. This is the main method of this class.

        The glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
        represents the relative position of the center of the glimpse on the image. delta determines
        the zoom factor of the glimpse.

        Args:
            images (B, h, w):  A batch of images
            glimpse_params (B, 3):  A batch of glimpse parameters (h_center, w_center, delta)

        Returns:
            (B, glimpse_h, glimpse_w): A batch of glimpses.

        """
        batch_size, image_h, image_w = images.size()
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, (2)], center_caps=glimpse_params[:, (0)], image_size=image_h, glimpse_size=self.glimpse_h)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, (2)], center_caps=glimpse_params[:, (1)], image_size=image_w, glimpse_size=self.glimpse_w)
        glimpses = images
        glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
        glimpses = torch.bmm(glimpses, F_w)
        return glimpses


class ARC(nn.Module):
    """
    This class implements the Attentive Recurrent Comparators. This module has two main parts.

    1.) controller: The RNN module that takes as input glimpses from a pair of images and emits a hidden state.

    2.) glimpser: A Linear layer that takes the hidden state emitted by the controller and generates the glimpse
                    parameters. These glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
                    represents the relative position of the center of the glimpse on the image. delta determines
                    the zoom factor of the glimpse.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, controller_out: int=128) ->None:
        super().__init__()
        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out
        self.controller = nn.LSTMCell(input_size=glimpse_h * glimpse_w, hidden_size=self.controller_out)
        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w)

    def forward(self, image_pairs: Variable) ->Variable:
        """
        The method calls the internal _forward() method which returns hidden states for all time steps. This i

        Args:
            image_pairs (B, 2, h, w):  A batch of pairs of images

        Returns:
            (B, controller_out): A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """
        all_hidden = self._forward(image_pairs)
        last_hidden = all_hidden[(-1), :, :]
        return last_hidden

    def _forward(self, image_pairs: Variable) ->Variable:
        """
        The main forward method of ARC. But it returns hidden state from all time steps (all glimpses) as opposed to
        just the last one. See the exposed forward() method.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (2*num_glimpses, B, controller_out) Hidden states from ALL time steps.

        """
        image_pairs = image_pairs.float()
        batch_size = image_pairs.size()[0]
        all_hidden = []
        Hx = Variable(torch.zeros(batch_size, self.controller_out))
        Cx = Variable(torch.zeros(batch_size, self.controller_out))
        if use_cuda:
            Hx, Cx = Hx, Cx
        for turn in range(2 * self.num_glimpses):
            images_to_observe = image_pairs[:, (turn % 2)]
            glimpse_params = torch.tanh(self.glimpser(Hx))
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)
            flattened_glimpses = glimpses.view(batch_size, -1)
            Hx, Cx = self.controller(flattened_glimpses, (Hx, Cx))
            all_hidden.append(Hx)
        all_hidden = torch.stack(all_hidden)
        return all_hidden


class ArcBinaryClassifier(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, controller_out: int=128):
        super().__init__()
        self.arc = ARC(num_glimpses=num_glimpses, glimpse_h=glimpse_h, glimpse_w=glimpse_w, controller_out=controller_out)
        self.dense1 = nn.Linear(controller_out, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image_pairs: Variable) ->Variable:
        arc_out = self.arc(image_pairs)
        d1 = F.elu(self.dense1(arc_out))
        decision = torch.sigmoid(self.dense2(d1))
        return decision

    def save_to_file(self, file_path: str) ->None:
        torch.save(self.state_dict(), file_path)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ARC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ArcBinaryClassifier,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_sanyam5_arc_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

