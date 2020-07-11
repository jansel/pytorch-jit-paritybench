import sys
_module = sys.modules[__name__]
del sys
LFWDataset = _module
TripletFaceDataset = _module
clean_msceleb_using_openface = _module
download_vgg_face_dataset = _module
extract_msceleb = _module
eval_metrics = _module
logger = _module
model = _module
train_center = _module
train_triplet = _module
utils = _module
vis = _module

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


from torchvision.models import resnet18


import math


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


from torch.nn.parameter import Parameter


import torch.optim as optim


import torchvision.transforms as transforms


from torchvision.datasets import ImageFolder


from torch.autograd import Function


import torch.backends.cudnn as cudnn


import collections


import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


from torchvision import datasets


from torchvision import transforms


import matplotlib.patheffects as PathEffects


class FaceModel(nn.Module):

    def __init__(self, embedding_size, num_classes, pretrained=False):
        super(FaceModel, self).__init__()
        self.model = resnet18(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(512 * 3 * 3, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        alpha = 10
        self.features = self.features * alpha
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


class FaceModelCenter(nn.Module):

    def __init__(self, embedding_size, num_classes, checkpoint=None):
        super(FaceModelCenter, self).__init__()
        self.model = resnet18()
        self.model.avgpool = None
        self.model.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.model.fc2 = nn.Linear(512, embedding_size)
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        self.num_classes = num_classes
        self.apply(self.weights_init)
        if checkpoint is not None:
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
                self.centers = checkpoint['centers']
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if 'classifier' not in name:
                        if isinstance(param, Parameter):
                            param = param.data
                        own_state[name].copy_(param)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def get_center_loss(self, target, alpha):
        batch_size = target.size(0)
        features_dim = self.features.size(1)
        target_expand = target.view(batch_size, 1).expand(batch_size, features_dim)
        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0, target_expand)
        criterion = nn.MSELoss()
        center_loss = criterion(self.features, centers_batch)
        diff = centers_batch - self.features
        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0, torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1, 1).expand(batch_size, features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-06)
        diff_cpu = alpha * diff_cpu
        for i in range(batch_size):
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())
        return center_loss, self.centers

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        x = self.model.fc2(x)
        self.features = x
        self.features_norm = self.l2_norm(x)
        return self.features_norm

    def forward_classifier(self, x):
        features_norm = self.forward(x)
        x = self.model.classifier(features_norm)
        return F.log_softmax(x)

