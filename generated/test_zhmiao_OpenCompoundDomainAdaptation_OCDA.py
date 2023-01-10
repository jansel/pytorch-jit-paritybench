import sys
_module = sys.modules[__name__]
del sys
main = _module
source = _module
extract_domain_factor_ftr = _module
test_cond_net = _module
train_domain_factor_net = _module
train_mann_net = _module
train_scheduled_mann_net = _module
train_source_net = _module
utils = _module
data = _module
mnist = _module
mnistm = _module
multipie = _module
sampler = _module
svhn_balanced = _module
synnum = _module
usps = _module
utils = _module
models = _module
amn_net = _module
cos_norm_classifier = _module
disc_centroids_loss = _module
domain_factor_backbone = _module
domain_factor_net = _module
mann_net = _module
resnet18 = _module
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


from copy import deepcopy


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data.sampler import Sampler


from torch.utils import data


from scipy.io import loadmat


import logging


from torchvision import transforms


import torch.nn as nn


import torchvision.models as models


import math


from torch.nn.parameter import Parameter


from torch.autograd.function import Function


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import ResNet


from torch.nn import init


def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()


class MemoryNet(nn.Module):
    num_channels = 3
    image_size = 32
    name = 'MemoryNet'
    """Basic class which does classification."""

    def __init__(self, num_cls=10, weights_init=None, feat_dim=512):
        super(MemoryNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_ctr = disc_centroids_loss.create_loss(feat_dim, num_cls)
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


def register_model(name):

    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


class AMNClassifier(MemoryNet):
    """Classifier used for SVHN source experiment"""
    num_channels = 3
    image_size = 32
    name = 'AMN'
    out_dim = 512

    def setup_net(self):
        self.conv_params = nn.Sequential(nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(64), nn.Dropout2d(0.1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(128), nn.Dropout2d(0.3), nn.ReLU(), nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU())
        self.fc_params = nn.Sequential(nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512))
        self.classifier = cos_norm_classifier.create_model(512, self.num_cls)

    def forward(self, x, with_ft=True):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        feat = x.clone()
        score = self.classifier(x)
        if with_ft:
            return score, feat
        else:
            return score


class AMNClassifier_res(MemoryNet):
    """Classifier used for face source experiment"""
    num_channels = 3
    image_size = 224
    name = 'AMN_res'
    out_dim = 512

    def setup_net(self):
        resnet18 = models.resnet18(pretrained=False)
        modules_resnet18 = list(resnet18.children())[:-1]
        self.feat_model = nn.Sequential(*modules_resnet18)
        self.classifier = cos_norm_classifier.create_model(512, self.num_cls)

    def forward(self, x, with_ft=True):
        x = self.feat_model(x)
        x = torch.squeeze(x)
        feat = x.clone()
        score = self.classifier(x)
        if with_ft:
            return score, feat
        else:
            return score


class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = norm_x / (1 + norm_x) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


class DiscCentroidsLossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())
        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centroids / batch_size, None


class DiscCentroidsLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's                              dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss_attract = self.disccentroidslossfunc(feat.clone(), label, self.centroids, batch_size_tensor).squeeze()
        distmat = torch.pow(feat.clone(), 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat.clone(), self.centroids.t())
        classes = torch.arange(self.num_classes).long()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1000000.0)
        loss = loss_attract + 0.01 * loss_repel
        return loss


class DomainFactorBackbone(nn.Module):

    def __init__(self):
        super(DomainFactorBackbone, self).__init__()
        self.num_channels = 3
        self.setup_net()

    def setup_net(self):
        self.conv_params = nn.Sequential(nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(64), nn.Dropout2d(0.1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(128), nn.Dropout2d(0.3), nn.ReLU(), nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU())
        self.fc_params = nn.Sequential(nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512))

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x


class Decoder(nn.Module):

    def __init__(self, input_dim=1024):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(nn.Linear(input_dim, 4096))
        self.decoder = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 3, kernel_size=1))

    def forward(self, x):
        assert x.size(1) == self.input_dim
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x


def get_model(name, **args):
    net = models[name](**args)
    if torch.cuda.is_available():
        net = net
    return net


class DomainFactorNet(nn.Module):
    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self, num_cls=10, base_model='LeNet', domain_factor_model='LeNet', content_weights_init=None, weights_init=None, eval=False, feat_dim=512):
        super(DomainFactorNet, self).__init__()
        self.name = 'DomainFactorNet'
        self.base_model = base_model
        self.domain_factor_model = domain_factor_model
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
        self.rec_criterion = nn.SmoothL1Loss()
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init, eval=eval)
        elif content_weights_init is not None:
            self.load_content_net(content_weights_init)
        else:
            raise Exception('MannNet must be initialized with weights.')

    def forward(self, x):
        pass

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)
        self.domain_factor_net = get_model(self.domain_factor_model)
        self.discriminator_cls = cos_norm_classifier.create_model(512, self.num_cls)
        self.decoder = Decoder(input_dim=1024)
        self.image_size = self.tgt_net.image_size
        self.num_channels = self.tgt_net.num_channels

    def load(self, init_path, eval=False):
        """
        Load weights from pretrained tgt model
        and initialize DomainFactorNet from pretrained tgr model.
        """
        net_init_dict = torch.load(init_path)
        None
        self.load_state_dict(net_init_dict, strict=False)
        load_keys = set(net_init_dict.keys())
        self_keys = set(self.state_dict().keys())
        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        None
        None
        if not eval:
            None
            tgt_weights = deepcopy(self.tgt_net.state_dict())
            self.domain_factor_net.load_state_dict(tgt_weights, strict=False)
            load_keys_sty = set(tgt_weights.keys())
            self_keys_sty = set(self.domain_factor_net.state_dict().keys())
            missing_keys_sty = self_keys_sty - load_keys_sty
            unused_keys_sty = load_keys_sty - self_keys_sty
            None
            None
        None
        self.discriminator_cls.weight.data = self.tgt_net.state_dict()['classifier.weight'].data.clone()

    def load_content_net(self, init_path):
        self.tgt_net.load(init_path)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_domain_factor_net(self, out_path):
        torch.save(self.domain_factor_net.state_dict(), out_path)


class MannNet(nn.Module):
    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self, num_cls=10, model='LeNet', src_weights_init=None, weights_init=None, use_domain_factor_selector=False, centroids_path=None, feat_dim=512):
        super(MannNet, self).__init__()
        self.name = 'MannNet'
        self.base_model = model
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.use_domain_factor_selector = use_domain_factor_selector
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
        self.centroids = torch.from_numpy(np.load(centroids_path)).float()
        assert self.centroids is not None
        self.centroids.requires_grad = False
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init)
        elif src_weights_init is not None:
            self.load_src_net(src_weights_init)
        else:
            raise Exception('MannNet must be initialized with weights.')

    def forward(self, x_s, x_t):
        """Pass source and target images through their respective networks."""
        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)
        if self.discrim_feat:
            d_s = self.discriminator(x_s.clone())
            d_t = self.discriminator(x_t.clone())
        else:
            d_s = self.discriminator(score_s.clone())
            d_t = self.discriminator(score_t.clone())
        return score_s, score_t, d_s, d_t

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.src_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls, feat_dim=self.feat_dim)
        input_dim = self.num_cls
        self.discriminator = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2))
        self.fc_selector = nn.Linear(self.feat_dim, self.feat_dim)
        if self.use_domain_factor_selector:
            self.domain_factor_selector = nn.Linear(self.feat_dim, self.feat_dim)
        self.classifier = cos_norm_classifier.create_model(self.feat_dim, self.num_cls)
        self.image_size = self.src_net.image_size
        self.num_channels = self.src_net.num_channels

    def load(self, init_path):
        """Loads full src and tgt models."""
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def load_src_net(self, init_path):
        """Initialize source and target with source weights."""
        self.src_net.load(init_path)
        self.tgt_net.load(init_path)
        net_init_dict = torch.load(init_path)
        classifier_weights = net_init_dict['classifier.weight']
        self.classifier.weight.data = classifier_weights.data.clone()

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CosNorm_Classifier,
     lambda: ([], {'in_dims': 4, 'out_dims': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_zhmiao_OpenCompoundDomainAdaptation_OCDA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

