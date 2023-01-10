import sys
_module = sys.modules[__name__]
del sys
dataloaders = _module
loader = _module
s3dis = _module
scannet = _module
main = _module
models = _module
attention = _module
dgcnn = _module
mpti = _module
mpti_learner = _module
proto_learner = _module
protonet = _module
collect_s3dis_data = _module
collect_scannet_data = _module
room2blocks = _module
runs = _module
eval = _module
fine_tune = _module
mpti_train = _module
pre_train = _module
proto_train = _module
utils = _module
checkpoint_util = _module
cuda_util = _module
logger = _module

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


import random


import math


import numpy as np


from itertools import combinations


import torch


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import copy


from torch import optim


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


class SelfAttention(nn.Module):

    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel
        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel
        self.temperature = self.out_channel ** 0.5
        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)
        k = self.k_map(x)
        v = self.v_map(x)
        attn = torch.matmul(q.transpose(1, 2) / self.temperature, k)
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1, 2))
        return y.transpose(1, 2)


class conv2d(nn.Module):

    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i == 0 else layer_dims[i - 1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class conv1d(nn.Module):

    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i == 0 else layer_dims[i - 1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_edge_feature(x, K=20, idx=None):
    """Construct edge feature for each point
      Args:
        x: point clouds (B, C, N)
        K: int
        idx: knn index, if not None, the shape is (B, N, K)
      Returns:
        edge feat: (B, 2C, N, K)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=K)
    central_feat = x.unsqueeze(-1).expand(-1, -1, -1, K)
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1).contiguous().view(B, C, N * K)
    knn_feat = torch.gather(x, dim=2, index=idx).contiguous().view(B, C, N, K)
    edge_feat = torch.cat((knn_feat - central_feat, central_feat), dim=1)
    return edge_feat


class DGCNN(nn.Module):
    """
    DGCNN with only stacked EdgeConv, return intermediate features if use attention
    Parameters:
      edgeconv_widths: list of layer widths of edgeconv blocks [[],[],...]
      mlp_widths: list of layer widths of mlps following Edgeconv blocks
      nfeat: number of input features
      k: number of neighbors
      conv_aggr: neighbor information aggregation method, Option:['add', 'mean', 'max', None]
    """

    def __init__(self, edgeconv_widths, mlp_widths, nfeat, k=20, return_edgeconvs=False):
        super(DGCNN, self).__init__()
        self.n_edgeconv = len(edgeconv_widths)
        self.k = k
        self.return_edgeconvs = return_edgeconvs
        self.edge_convs = nn.ModuleList()
        for i in range(self.n_edgeconv):
            if i == 0:
                in_feat = nfeat * 2
            else:
                in_feat = edgeconv_widths[i - 1][-1] * 2
            self.edge_convs.append(conv2d(in_feat, edgeconv_widths[i]))
        in_dim = 0
        for edgeconv_width in edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.conv = conv1d(in_dim, mlp_widths)

    def forward(self, x):
        edgeconv_outputs = []
        for i in range(self.n_edgeconv):
            x = get_edge_feature(x, K=self.k)
            x = self.edge_convs[i](x)
            x = x.max(dim=-1, keepdim=False)[0]
            edgeconv_outputs.append(x)
        out = torch.cat(edgeconv_outputs, dim=1)
        out = self.conv(out)
        if self.return_edgeconvs:
            return edgeconv_outputs, out
        else:
            return edgeconv_outputs[0], out


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()
        self.num_convs = len(params)
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i - 1]
            self.convs.append(nn.Sequential(nn.Conv1d(in_dim, params[i], 1), nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs - 1:
                x = F.relu(x)
        return x


class MultiPrototypeTransductiveInference(nn.Module):

    def __init__(self, args):
        super(MultiPrototypeTransductiveInference, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.n_subprototypes = args.n_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.sigma
        self.n_classes = self.n_way + 1
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)
        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \\in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, self.feat_dim, self.n_points)
        query_feat = self.getFeatures(query_x)
        query_feat = query_feat.transpose(1, 2).contiguous().view(-1, self.feat_dim)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)
        fg_prototypes, fg_labels = self.getForegroundPrototypes(support_feat, fg_mask, k=self.n_subprototypes)
        bg_prototype, bg_labels = self.getBackgroundPrototypes(support_feat, bg_mask, k=self.n_subprototypes)
        if bg_prototype is not None and bg_labels is not None:
            prototypes = torch.cat((bg_prototype, fg_prototypes), dim=0)
            prototype_labels = torch.cat((bg_labels, fg_labels), dim=0)
        else:
            prototypes = fg_prototypes
            prototype_labels = fg_labels
        self.num_prototypes = prototypes.shape[0]
        self.num_nodes = self.num_prototypes + query_feat.shape[0]
        Y = torch.zeros(self.num_nodes, self.n_classes)
        Y[:self.num_prototypes] = prototype_labels
        node_feat = torch.cat((prototypes, query_feat), dim=0)
        A = self.calculateLocalConstrainedAffinity(node_feat, k=self.k_connect)
        Z = self.label_propagate(A, Y)
        query_pred = Z[self.num_prototypes:, :]
        query_pred = query_pred.view(-1, query_y.shape[1], self.n_classes).transpose(1, 2)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)
        return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMutiplePrototypes(self, feat, k):
        """
        Extract multiple prototypes by points separation and assembly

        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
        """
        n = feat.shape[0]
        assert n > 0
        ratio = k / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
            num_prototypes = len(fps_index)
            farthest_seeds = feat[fps_index]
            distances = F.pairwise_distance(feat[..., None], farthest_seeds.transpose(0, 1)[None, ...], p=2)
            assignments = torch.argmin(distances, dim=1)
            prototypes = torch.zeros((num_prototypes, self.feat_dim))
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                prototypes[i] = selected.mean(0)
            return prototypes
        else:
            return feat

    def getForegroundPrototypes(self, feats, masks, k=100):
        """
        Extract foreground prototypes for each class via clustering point features within that class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
        """
        prototypes = []
        labels = []
        for i in range(self.n_way):
            feat = feats[i, ...].transpose(1, 2).contiguous().view(-1, self.feat_dim)
            index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1)
            feat = feat[index]
            class_prototypes = self.getMutiplePrototypes(feat, k)
            prototypes.append(class_prototypes)
            class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
            class_labels[:, i + 1] = 1
            labels.append(class_labels)
        prototypes = torch.cat(prototypes, dim=0)
        labels = torch.cat(labels, dim=0)
        return prototypes, labels

    def getBackgroundPrototypes(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
        """
        feats = feats.transpose(2, 3).contiguous().view(-1, self.feat_dim)
        index = torch.nonzero(masks.view(-1)).squeeze(1)
        feat = feats[index]
        if feat.shape[0] != 0:
            prototypes = self.getMutiplePrototypes(feat, k)
            labels = torch.zeros(prototypes.shape[0], self.n_classes)
            labels[:, 0] = 1
            return prototypes, labels
        else:
            return None, None

    def calculateLocalConstrainedAffinity(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        X = node_feat.detach().cpu().numpy()
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:])
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.num_nodes, k, self.feat_dim)
        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:, None, :], knn_feat, dim=2)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:, :, None], knn_feat.transpose(1, 2), p=2)
            knn_similarity = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        A = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float)
        A = A.scatter_(1, I, knn_similarity)
        A = A + A.transpose(0, 1)
        identity_matrix = torch.eye(self.num_nodes, requires_grad=False)
        A = A * (1 - identity_matrix)
        return A

    def label_propagate(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        eps = np.finfo(float).eps
        D = A.sum(1)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv)
        S = D_sqrt_inv @ A @ D_sqrt_inv
        Z = torch.inverse(torch.eye(self.num_nodes) - alpha * S + eps) @ Y
        return Z

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)


class ProtoNet(nn.Module):

    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \\in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)
        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
        query_pred = torch.stack(similarity, dim=1)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)
        return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-05)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [(fg_feat[way, ...].sum(dim=0) / self.k_shot) for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = -F.pairwise_distance(feat, prototype[None, ..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)


class DGCNNSeg(nn.Module):

    def __init__(self, args, num_classes):
        super(DGCNNSeg, self).__init__()
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.segmenter = nn.Sequential(nn.Conv1d(in_dim, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Conv1d(128, num_classes, 1))

    def forward(self, pc):
        num_points = pc.shape[2]
        edgeconv_feats, point_feat = self.encoder(pc)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0]
        edgeconv_feats.append(global_feat.expand(-1, -1, num_points))
        pc_feat = torch.cat(edgeconv_feats, dim=1)
        logits = self.segmenter(pc_feat)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseLearner,
     lambda: ([], {'in_channels': 4, 'params': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'in_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (conv1d,
     lambda: ([], {'in_feat': 4, 'layer_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (conv2d,
     lambda: ([], {'in_feat': 4, 'layer_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Na_Z_attMPTI(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

