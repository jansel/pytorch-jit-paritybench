import sys
_module = sys.modules[__name__]
del sys
setup = _module
torchbench = _module
datasets = _module
ade20k = _module
camvid = _module
cityscapes = _module
coco = _module
conll03 = _module
pascalcontext = _module
utils = _module
wikitext103 = _module
image_classification = _module
cifar10 = _module
cifar100 = _module
imagenet = _module
mnist = _module
stl10 = _module
svhn = _module
image_generation = _module
utils = _module
language_modelling = _module
utils = _module
object_detection = _module
coco = _module
coco_eval = _module
pascalvoc = _module
transforms = _module
utils = _module
voc_eval = _module
semantic_segmentation = _module
common_utils = _module
fakedata_generation = _module
test_datasets = _module
utils = _module
version = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch.utils.data import DataLoader


import torchvision.datasets as datasets


import torchvision.transforms as transforms


import numpy as np


from scipy import linalg


from scipy.stats import entropy


import torch


import torch.utils.data


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


from torchvision.models.inception import inception_v3


from torch.nn import CrossEntropyLoss


import torchvision


import time


class FIDInceptionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True)
        self.inception_model.Mixed_7c.register_forward_hook(self.output_hook)
        self.mixed_7c_output = None

    def output_hook(self, module, input, output):
        """Output will be of dimensions (batch_size, 2048, 8, 8)."""
        self.mixed_7c_output = output

    def forward(self, x):
        """x inputs should be (N, 3, 299, 299) in range -1 to 1.

        Returns activations in form of torch.tensor of shape (N, 2048, 1, 1)
        """
        self.inception_model(x)
        activations = self.mixed_7c_output
        activations = F.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


class InceptionScore(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True)
        self.up = nn.Upsample(size=(299, 299), mode='bilinear')

    def forward(self, x):
        """x inputs should be (N, 3, 299, 299) in range -1 to 1.

        Returns class probabilities in form of torch.tensor of shape
        (N, 1000, 1, 1).
        """
        x = self.up(x)
        x = self.inception_model(x)
        return F.softmax(x).data.cpu().numpy()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_paperswithcode_torchbench(_paritybench_base):
    pass
