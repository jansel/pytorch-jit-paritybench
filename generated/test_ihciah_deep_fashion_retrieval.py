import sys
_module = sys.modules[__name__]
del sys
master = _module
config = _module
data = _module
debug = _module
feaure_extractor = _module
in_shop_eval = _module
kmeans = _module
net = _module
retrieval = _module
scripts = _module
category_count = _module
model_convertor = _module
train = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.optim as optim


from torchvision import transforms


from torch.autograd import Variable


import torchvision


import numpy as np


from scipy import spatial


import torch.nn.functional as F


import random


import time


CATEGORIES = 20


INTER_DIM = 512


DATASET_BASE = '/DATACETNER/1/ch/deepfashion_data'


def load_model(path=None):
    if not path:
        return None
    full = os.path.join(DATASET_BASE, 'models', path)
    for i in [path, full]:
        if os.path.isfile(i):
            return torch.load(i)
    return None


class f_model(nn.Module):
    """
    input: N * 3 * 224 * 224
    output: N * num_classes, N * inter_dim, N * C' * 7 * 7
    """

    def __init__(self, freeze_param=False, inter_dim=INTER_DIM, num_classes
        =CATEGORIES, model_path=None):
        super(f_model, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        state_dict = self.backbone.state_dict()
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in
            model_dict})
        self.backbone.load_state_dict(model_dict)
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.avg_pooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes)
        state = load_model(model_path)
        if state:
            new_state = self.state_dict()
            new_state.update({k: v for k, v in state.items() if k in new_state}
                )
            self.load_state_dict(new_state)

    def forward(self, x):
        x = self.backbone(x)
        pooled = self.avg_pooling(x)
        inter_out = self.fc(pooled.view(pooled.size(0), -1))
        out = self.fc2(inter_out)
        return out, inter_out, x


class c_model(nn.Module):
    """
    input: N * C * 224 * 224
    output: N * C * 7 * 7
    """

    def __init__(self, pooling_size=32):
        super(c_model, self).__init__()
        self.pooling = nn.AvgPool2d(pooling_size)

    def forward(self, x):
        return self.pooling(x)


class p_model(nn.Module):
    """
    input: N * C * W * H
    output: N * 1 * W * H
    """

    def __init__(self):
        super(p_model, self).__init__()

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(x, c)
        return pooled.view(n, 1, w, h)


COLOR_TOP_N = 10


class FeatureExtractor(nn.Module):

    def __init__(self, deep_module, color_module, pooling_module):
        super(FeatureExtractor, self).__init__()
        self.deep_module = deep_module
        self.color_module = color_module
        self.pooling_module = pooling_module
        self.deep_module.eval()
        self.color_module.eval()
        self.pooling_module.eval()

    def forward(self, x):
        cls, feat, conv_out = self.deep_module(x)
        color = self.color_module(x).cpu().data.numpy()
        weight = self.pooling_module(conv_out).cpu().data.numpy()
        result = []
        for i in range(cls.size(0)):
            weight_n = weight[i].reshape(-1)
            idx = np.argpartition(weight_n, -COLOR_TOP_N)[-COLOR_TOP_N:][::-1]
            color_n = color[i].reshape(color.shape[1], -1)
            color_selected = color_n[:, (idx)].reshape(-1)
            result.append(color_selected)
        return feat.cpu().data.numpy(), result


class TripletMarginLossCosine(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletMarginLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_p = 1 - F.cosine_similarity(anchor, positive).view(-1, 1)
        d_n = 1 - F.cosine_similarity(anchor, negative).view(-1, 1)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ihciah_deep_fashion_retrieval(_paritybench_base):
    pass
    def test_000(self):
        self._check(TripletMarginLossCosine(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(c_model(*[], **{}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(p_model(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

