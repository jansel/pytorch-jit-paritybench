import sys
_module = sys.modules[__name__]
del sys
combine = _module
setup = _module
dataset = _module
model = _module
test = _module
train = _module
train_options = _module
transforms = _module

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
xrange = range
wraps = functools.wraps


import random


import numpy as np


import torch


import torch.utils.data as data


from torch import nn


import torchvision


import time


import torch.nn.parallel


import torch.optim


import torch.backends.cudnn as cudnn


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def resize_mv(img, shape, interpolation):
    return np.stack([cv2.resize(img[..., i], shape, interpolation) for i in range(2)], axis=2)


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=False, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):
        im_size = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img[offset_w:offset_w + crop_w, offset_h:offset_h + crop_h] for img in img_group]
        if crop_img_group[0].shape[2] == 3:
            ret_img_group = [cv2.resize(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR) for img in crop_img_group]
        elif crop_img_group[0].shape[2] == 2:
            ret_img_group = [resize_mv(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR) for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [(self.input_size[1] if abs(x - self.input_size[1]) < 3 else x) for x in crop_sizes]
        crop_w = [(self.input_size[0] if abs(x - self.input_size[0]) < 3 else x) for x in crop_sizes]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))
            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret


class GroupRandomHorizontalFlip(object):

    def __init__(self, is_mv=False):
        self._is_mv = is_mv

    def __call__(self, img_group, is_mv=False):
        if random.random() < 0.5:
            ret = [img[:, ::-1, :].astype(np.int32) for img in img_group]
            if self._is_mv:
                for i in range(len(ret)):
                    ret[i] -= 128
                    ret[i][..., 0] *= -1
                    ret[i] += 128
            return ret
        else:
            return img_group


class Model(nn.Module):

    def __init__(self, num_class, num_segments, representation, base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        None
        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        if self._representation == 'mv':
            setattr(self.base_model, 'conv1', nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        input = input.view((-1,) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)
        base_out = self.base_model(input)
        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, 0.875, 0.75]
        else:
            scales = [1, 0.875, 0.75, 0.66]
        None
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self._input_size, scales), GroupRandomHorizontalFlip(is_mv=self._representation == 'mv')])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Model,
     lambda: ([], {'num_class': 4, 'num_segments': 4, 'representation': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_chaoyuaw_pytorch_coviar(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

