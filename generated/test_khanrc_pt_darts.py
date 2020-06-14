import sys
_module = sys.modules[__name__]
del sys
architect = _module
augment = _module
config = _module
genotypes = _module
augment_cells = _module
augment_cnn = _module
ops = _module
search_cells = _module
search_cnn = _module
preproc = _module
benchmark = _module
dot2png = _module
search = _module
utils = _module
visualize = _module

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


import numpy as np


from collections import namedtuple


import torch.nn.functional as F


from torch.nn.parallel._functions import Broadcast


import logging


import copy


import time


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """

    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat
        self.dag = gt.to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        return s_out


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=input_size - 5, padding=0, count_include_pad=False), nn.
            Conv2d(C, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Conv2d(128, 768, kernel_size=2, bias=
            False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class AugmentCNN(nn.Module):
    """ Augmented CNN model """

    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary,
        genotype, stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=
            False), nn.BatchNorm2d(C_cur))
        C_pp, C_p, C_cur = C_cur, C_cur, C
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p,
                reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out
            if i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size // 4, C_p, n_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x


class DropPath_(nn.Module):

    def __init__(self, p=0.0):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)
        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True
        ):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding,
                count_include_pad=False)
        else:
            raise ValueError()
        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(C_in, C_out,
            kernel_size, stride, padding, bias=False), nn.BatchNorm2d(C_out,
            affine=affine))

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True
        ):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(C_in, C_in, (
            kernel_length, 1), stride, padding, bias=False), nn.Conv2d(C_in,
            C_out, (1, kernel_length), stride, padding, bias=False), nn.
            BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        affine=True):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(C_in, C_in,
            kernel_size, stride, padding, dilation=dilation, groups=C_in,
            bias=False), nn.Conv2d(C_in, C_out, 1, stride=1, padding=0,
            bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(DilConv(C_in, C_in, kernel_size, stride,
            padding, dilation=1, affine=affine), DilConv(C_in, C_out,
            kernel_size, 1, padding, dilation=1, affine=affine))

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        return x[:, :, ::self.stride, ::self.stride] * 0.0


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda
    C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1,
    affine=affine), 'skip_connect': lambda C, stride, affine: Identity() if
    stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3':
    lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2,
    affine=affine), 'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C,
    7, stride, 3, affine=affine), 'dil_conv_3x3': lambda C, stride, affine:
    DilConv(C, C, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda C,
    stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3,
    affine=affine)}


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(
                states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], dim=1)
        return s_out


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4,
        stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=
            False), nn.BatchNorm2d(C_cur))
        C_pp, C_p, C_cur = C_cur, C_cur, C
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction
                )
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))
        ]
    return l_copies


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4,
        stem_multiplier=3, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        n_ops = len(gt.PRIMITIVES)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(0.001 * torch.randn(i + 2,
                n_ops)))
            self.alpha_reduce.append(nn.Parameter(0.001 * torch.randn(i + 2,
                n_ops)))
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes,
            stem_multiplier)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.
            alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.
            alpha_reduce]
        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)
        xs = nn.parallel.scatter(x, self.device_ids)
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs,
            wnormal_copies, wreduce_copies)), devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter('%(message)s'))
        logger.info('####### ALPHA #######')
        logger.info('# Alpha - normal')
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info('\n# Alpha - reduce')
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info('#####################')
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)
        return gt.Genotype(normal=gene_normal, normal_concat=concat, reduce
            =gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_khanrc_pt_darts(_paritybench_base):
    pass
    def test_000(self):
        self._check(DilConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DropPath_(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(FacConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_length': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FactorizedReduce(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SepConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(StdConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Zero(*[], **{'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

