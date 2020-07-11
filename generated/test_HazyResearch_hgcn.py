import sys
_module = sys.modules[__name__]
del sys
master = _module
config = _module
layers = _module
att_layers = _module
hyp_layers = _module
layers = _module
manifolds = _module
base = _module
euclidean = _module
hyperboloid = _module
poincare = _module
models = _module
base_models = _module
decoders = _module
encoders = _module
optimizers = _module
radam = _module
train = _module
utils = _module
data_utils = _module
eval_utils = _module
hyperbolicity = _module
math_utils = _module
train_utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.nn.init as init


from torch.nn.modules.module import Module


from torch.nn.parameter import Parameter


from torch.nn import Parameter


from sklearn.metrics import roc_auc_score


from sklearn.metrics import average_precision_score


from torch.optim import Adam


import torch.optim


import logging


import time


import scipy.sparse as sp


import torch.nn.modules.loss


class DenseAtt(nn.Module):

    def __init__(self, in_features, dropout, act):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.act = act
        self.in_features = in_features

    def forward(self, x, adj):
        n = x.size(0)
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)
        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[(0), :] * ctx.N + a._indices()[(1), :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):

    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[(edge[(0), :]), :], h[(edge[(1), :]), :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [SpGraphAttentionLayer(input_dim, output_dim, dropout=dropout, alpha=alpha, activation=activation) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return h, adj


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(self.c_in, self.c_out)


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(self.in_features, self.out_features, self.c)


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(self.in_features, self.out_features)


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1.0 / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c
        else:
            self.c = nn.Parameter(torch.Tensor([1.0]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = x, adj
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(self.input_dim, self.output_dim, self.bias, self.c)


model2decoder = {'GCN': GCNDecoder, 'GAT': GATDecoder, 'HNN': LinearDecoder, 'HGCN': LinearDecoder, 'MLP': LinearDecoder, 'Shallow': LinearDecoder}


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1.0, 1.0 / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.0] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1['f1'] < m2['f1']


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[(idx[:, (0)]), :]
        emb_out = h[(idx[:, (1)]), :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = x, adj
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + [args.dim] * (args.num_layers - 1)
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(hyp_layers.HNNLayer(self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(hyp_layers.HyperbolicGraphConvolution(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, 'The embeddings you passed seem to be for another dataset.'
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[(self.all_nodes), :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FermiDiracDecoder,
     lambda: ([], {'r': 4, 't': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'dropout': 0.5, 'act': _mock_layer(), 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_HazyResearch_hgcn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

