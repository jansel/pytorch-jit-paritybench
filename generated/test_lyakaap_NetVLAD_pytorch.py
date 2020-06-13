import sys
_module = sys.modules[__name__]
del sys
hard_triplet_loss = _module
netvlad = _module

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


import torch.nn.functional as F


def _pairwise_distance(x, squared=False, eps=1e-16):
    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)
    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def _get_anchor_negative_triplet_mask(labels):
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1
    return mask


def _get_anchor_positive_triplet_mask(labels):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = indices_not_equal * labels_equal
    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(
        labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)
    mask = distinct_indices * valid_labels
    return mask


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)
        if self.hardest:
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels
                ).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1,
                keepdim=True)
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels
                ).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1,
                keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=
                1, keepdim=True)
            triplet_loss = F.relu(hardest_positive_dist -
                hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)
            loss = anc_pos_dist - anc_neg_dist + self.margin
            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask
            triplet_loss = F.relu(triplet_loss)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)
            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16
                )
        return triplet_loss


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
        normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids)
            .unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(
            1, 0, 2, 3) - self.centroids.expand(x_flatten.size(-1), -1, -1
            ).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class EmbedNet(nn.Module):

    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):

    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lyakaap_NetVLAD_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(NetVLAD(*[], **{}), [torch.rand([4, 128, 64, 64])], {})

