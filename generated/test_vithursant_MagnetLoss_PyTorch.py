import sys
_module = sys.modules[__name__]
del sys
datasets = _module
fashion = _module
load_dataset = _module
magnet_loss = _module
magnet_loss = _module
magnet_tools = _module
utils = _module
magnet_loss_test = _module
models = _module
lenet = _module
vgg = _module
average_meter = _module
sampler = _module
visualizer = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch import optim


import numpy


import numpy as np


from math import ceil


import torch.nn.init as init


def expand_dims(var, dim=0):
    """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
        var = torch.range(0, 9).view(-1, 2)
        torch.expand_dims(var, 0).size()
        # (1, 5, 2)
    """
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)


def comparison_mask(a_labels, b_labels):
    """Computes boolean mask for distance comparisons"""
    return torch.eq(expand_dims(a_labels, 1), expand_dims(b_labels, 0))


def compute_euclidean_distance(x, y):
    return torch.sum((x - y) ** 2, dim=2)


def dynamic_partition(X, partitions, n_clusters):
    """Partitions the data into the number of cluster bins"""
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin


class MagnetLoss(nn.Module):
    """
    Magnet loss technique presented in the paper:
    ''Metric Learning with Adaptive Density Discrimination'' by Oren Rippel, Manohar Paluri, Piotr Dollar, Lubomir Bourdev in
    https://research.fb.com/wp-content/uploads/2016/05/metric-learning-with-adaptive-density-discrimination.pdf?

    Args:
        r: A batch of features.
        classes: Class labels for each example.
        clusters: Cluster labels for each example.
        cluster_classes: Class label for each cluster.
        n_clusters: Total number of clusters.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """

    def __init__(self, alpha=1.0):
        super(MagnetLoss, self).__init__()
        self.r = None
        self.classes = None
        self.clusters = None
        self.cluster_classes = None
        self.n_clusters = None
        self.alpha = alpha

    def forward(self, r, classes, m, d, alpha=1.0):
        GPU_INT_DTYPE = torch.cuda.IntTensor
        GPU_LONG_DTYPE = torch.cuda.LongTensor
        GPU_FLOAT_DTYPE = torch.cuda.FloatTensor
        self.r = r
        self.classes = torch.from_numpy(classes).type(GPU_LONG_DTYPE)
        self.clusters, _ = torch.sort(torch.arange(0, float(m)).repeat(d))
        self.clusters = self.clusters.type(GPU_INT_DTYPE)
        self.cluster_classes = self.classes[0:m * d:d]
        self.n_clusters = m
        self.alpha = alpha
        cluster_examples = dynamic_partition(self.r, self.clusters, self.
            n_clusters)
        cluster_means = torch.stack([torch.mean(x, dim=0) for x in
            cluster_examples])
        sample_costs = compute_euclidean_distance(cluster_means,
            expand_dims(r, 1))
        clusters_tensor = self.clusters.type(GPU_FLOAT_DTYPE)
        n_clusters_tensor = torch.arange(0, self.n_clusters).type(
            GPU_FLOAT_DTYPE)
        intra_cluster_mask = Variable(comparison_mask(clusters_tensor,
            n_clusters_tensor).type(GPU_FLOAT_DTYPE))
        intra_cluster_costs = torch.sum(intra_cluster_mask * sample_costs,
            dim=1)
        N = r.size()[0]
        variance = torch.sum(intra_cluster_costs) / float(N - 1)
        var_normalizer = -1 / (2 * variance ** 2)
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha
            )
        classes_tensor = self.classes.type(GPU_FLOAT_DTYPE)
        cluster_classes_tensor = self.cluster_classes.type(GPU_FLOAT_DTYPE)
        diff_class_mask = Variable(comparison_mask(classes_tensor,
            cluster_classes_tensor).type(GPU_FLOAT_DTYPE))
        diff_class_mask = 1 - diff_class_mask
        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)
        epsilon = 1e-08
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) +
            epsilon))
        total_loss = torch.mean(losses)
        return total_loss, losses


class LeNet(nn.Module):

    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        """
		Define the initialization function of LeNet, this function defines
		the basic structure of the neural network
		"""
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(64 * 7 * 7, self.emb_dim)
        self.layer1 = None
        self.layer2 = None
        self.features = None
        self.embeddings = None
        self.norm_embeddings = None

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        self.layer1 = x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        self.layer2 = x
        x = x.view(-1, self.num_flat_features(x))
        self.features = x
        x = self.emb(x)
        embeddings = x
        return embeddings, self.features

    def num_flat_features(self, x):
        """
		Calculate the total tensor x feature amount
		"""
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def name(self):
        return 'lenet-magnet'


class VGG(nn.Module):

    def __init__(self, depth, num_classes=10, channels=3):
        assert depth in cfg, 'Error: model depth invalid or undefined!'
        super(VGG, self).__init__()
        self.feature_extractor = self._make_layers(cfg[depth], channels)
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.classifier(x)
        return x, features

    def _make_layers(self, config, channels):
        layers = []
        in_channels = channels
        for x_cfg in config:
            if x_cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x_cfg, kernel_size=3,
                    padding=1), nn.BatchNorm2d(x_cfg), nn.ReLU(inplace=True)]
                in_channels = x_cfg
        return nn.Sequential(*layers)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_vithursant_MagnetLoss_PyTorch(_paritybench_base):
    pass
