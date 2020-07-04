import sys
_module = sys.modules[__name__]
del sys
flops_counter = _module
utils = _module
macro_decoder = _module
macro_genotypes = _module
macro_models = _module
micro_genotypes = _module
micro_models = _module
micro_operations = _module
cifar10_search = _module
evolution_search = _module
macro_encoding = _module
micro_encoding = _module
nsganet = _module
train_search = _module
test = _module
train = _module
macro_visualize = _module
micro_visualize = _module

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


import torch.nn as nn


import torch


import numpy as np


from copy import copy


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


import logging


import torch.utils


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import time


import torchvision


import torch.optim as optim


class Decoder(ABC):
    """
    Abstract genome decoder class.
    """

    @abstractmethod
    def __init__(self, list_genome):
        """
        :param list_genome: genome represented as a list.
        """
        self._genome = list_genome

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


class HourGlassDecoder(Decoder):
    """
    Decoder that deals with HourGlass-type networks.
    """

    def __init__(self, genome, n_stacks, out_feature_maps):
        """
        Constructor.
        :param genome: list, list of ints.
        :param n_stacks: int, number of hourglasses to use.
        :param out_feature_maps: int, number of output feature maps.
        """
        super().__init__(genome)
        self.n_stacks = n_stacks
        self.out_feature_maps = out_feature_maps

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def check_genome(genome):
        raise NotImplementedError()


class LOSComputationGraph:
    """
    Graph to hold information about the computation going on in
    """


    class Node:
        """
        Node to hold information.
        """

        def __init__(self, resolution, idx, residual=False):
            """
            Constructor.
            :param resolution: int, the resolution of the image at this point.
            :param idx: int, the index of the node in the graph (feed-forward, so this is ok).
            :param residual: bool, true if output of this node is needed at some point later in the graph.
            """
            self.resolution, self.idx, self.residual = (resolution, idx,
                residual)
            self.residual_node = None

        def __repr__(self):
            residual_str = ', saves residual' if self.residual else ''
            return '<Node index: {} resolution: {}'.format(self.idx, self.
                resolution) + residual_str + '>'

        def __str__(self):
            return self.__repr__()

        def __lt__(self, other):
            assert isinstance(other, LOSComputationGraph.Node)
            return self.idx < other.idx

    def __init__(self, genome, under_connect=True):
        """
        Make the computation graph specified by the genoms.
        :param genome: list, list of ints representing a genome.
        """
        self.graph = LOSComputationGraph.make_graph(genome, under_connect)

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return self.graph.__iter__()

    def items(self):
        return self.graph.items()

    def keys(self):
        return self.graph.keys()

    def values(self):
        return self.graph.values()

    def get_residual(self, node):
        """
        Determines if a particular node in the graph gets a residual connection.
        :param node: LOSComputationGraph.Node.
        :return: LOSComputationGraph.Node | None
        """
        if node in self.graph:
            for dep in self.graph[node]:
                if dep.resolution == node.resolution and dep.residual:
                    return dep
        return None

    @staticmethod
    def make_graph(genome, under_connect=True):
        """
        Make the computation graph.
        The is not exactly an adjacency list... The normal forward path through the network is as expected, but the
            skip connections are only listed in the receiving nodes, rather than the sending nodes.
            This makes things much easier when actually forward propagating.
        :param genome: list, list of ints representing a genome.
        :param under_connect: bool, if false, we will not allow "under connections".
            Where an under connection connects nodes that may occur below the current path. Ex:
            | X ----->  X ----->  X
            |   X --> X .. X --> X
            |     X  ......  X
            Where arrows are the normal residual connections and the dots are the optional under connections.
        :return: OrderedDict, dict of lists, adjacency list describing the computation graph.
        """
        adj = OrderedDict()
        nodes = [LOSComputationGraph.Node(pow(2, -(gene - 1)), i) for i,
            gene in enumerate(genome)]
        for i, (gene_i, gene_ipo) in enumerate(zip(nodes, nodes[1:])):
            adj[gene_i] = [gene_ipo]
        adj[nodes[-1]] = []
        previous_resolutions = {}
        previous_node = nodes[0]
        for node, adj_list in adj.items():
            if node.resolution in previous_resolutions:
                if (previous_node.resolution < node.resolution or 
                    previous_node.resolution > node.resolution and
                    under_connect):
                    previous_resolutions[node.resolution].residual = True
                    node.residual_node = previous_resolutions[node.resolution]
                    previous_resolutions[node.resolution] = node
                else:
                    previous_resolutions[node.resolution] = node
            else:
                previous_resolutions[node.resolution] = node
            previous_node = node
        return adj


class LOSHourGlassDecoder(HourGlassDecoder, nn.Module):
    """
    Line of sight HourGlass decoder.
    """
    STEP_TOLERANCE = 2
    GENE_LB = 0
    GENE_UB = 6

    def __init__(self, genome, n_stacks, out_feature_maps,
        pre_hourglass_channels=32, hourglass_channels=64):
        """
        Constructor.
        :param genome: list, list of ints satisfying properties defined in self.valid_genome.
        :param n_stacks: int, number of hourglasses to use.
        :param out_feature_maps, int, number of output feature maps.
        """
        HourGlassDecoder.__init__(self, genome, n_stacks, out_feature_maps)
        nn.Module.__init__(self)
        self.pre_hourglass_channels = pre_hourglass_channels
        self.hourglass_channels = hourglass_channels
        self.check_genome(genome)
        self.initial = nn.Sequential(nn.Conv2d(3, self.
            pre_hourglass_channels, kernel_size=7, stride=2, padding=3,
            bias=True), nn.BatchNorm2d(self.pre_hourglass_channels), nn.
            ReLU(inplace=True), HourGlassResidual(self.
            pre_hourglass_channels, self.pre_hourglass_channels))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.secondary = nn.Sequential(HourGlassResidual(self.
            pre_hourglass_channels, self.pre_hourglass_channels),
            HourGlassResidual(self.pre_hourglass_channels, self.
            hourglass_channels))
        graph = LOSComputationGraph(genome)
        hg_channels = self.hourglass_channels * LOSHourGlassBlock.EXPANSION
        hourglasses = [LOSHourGlassBlock(graph, self.hourglass_channels,
            hg_channels)]
        first_lin = [Lin(hg_channels, hg_channels)]
        second_lin = [Lin(hg_channels, self.hourglass_channels)]
        to_score_map = [nn.Conv2d(self.hourglass_channels, out_feature_maps,
            kernel_size=1, bias=True)]
        from_score_map = [nn.Conv2d(out_feature_maps, self.
            hourglass_channels + self.pre_hourglass_channels, kernel_size=1,
            bias=True)]
        skip_convs = [nn.Conv2d(self.hourglass_channels + self.
            pre_hourglass_channels, self.hourglass_channels + self.
            pre_hourglass_channels, kernel_size=1, bias=True)]
        skip_channels = self.pre_hourglass_channels
        for i in range(1, n_stacks):
            hourglasses.append(LOSHourGlassBlock(graph, self.
                hourglass_channels + skip_channels, hg_channels))
            first_lin.append(Lin(hg_channels, hg_channels))
            to_score_map.append(nn.Conv2d(self.hourglass_channels,
                out_feature_maps, kernel_size=1, bias=True))
            second_lin.append(Lin(hg_channels, self.hourglass_channels))
            if i < n_stacks - 1:
                skip_convs.append(nn.Conv2d(hg_channels, hg_channels,
                    kernel_size=1, bias=True))
                from_score_map.append(nn.Conv2d(out_feature_maps,
                    hg_channels, kernel_size=1, bias=True))
            skip_channels = self.hourglass_channels
        self.hourglasses = nn.ModuleList(hourglasses)
        self.first_lin = nn.ModuleList(first_lin)
        self.to_score_map = nn.ModuleList(to_score_map)
        self.from_score_map = nn.ModuleList(from_score_map)
        self.second_lin = nn.ModuleList(second_lin)
        self.skip_convs = nn.ModuleList(skip_convs)

    @staticmethod
    def check_genome(genome):
        """
        Make sure the genome is valid.
        :param genome: list, list of ints, representing the genome.
        :raises AssertionError: if genome is not valid.
        """
        assert isinstance(genome[0], int
            ), 'Genome should be a list of integers.'
        for gene in genome:
            assert LOSHourGlassDecoder.GENE_LB < gene < LOSHourGlassDecoder.GENE_UB, '{} is an invalid gene value, must be in range [{}, {}]'.format(
                gene, LOSHourGlassDecoder.GENE_LB, LOSHourGlassDecoder.GENE_UB)
        for i in range(len(genome) - 1):
            step = abs(genome[i] - genome[i + 1])
            assert step <= LOSHourGlassDecoder.STEP_TOLERANCE, 'Attempted to step {} resolutions, cannot step more than 2 resolutions.'.format(
                step)

    def get_model(self):
        """
        In other decoders, we'd return a module object, but since self is an nn.Module, we return self.
        :return: self
        """
        return self

    def forward(self, x):
        """
        Forward operation.
        :param x: Variable, input
        :return: list, list of Variables, intermediate and final score maps.
        """
        maps = []
        x = self.initial(x)
        x = self.pool(x)
        skip = x.clone()
        x = self.secondary(x)
        for i in range(self.n_stacks):
            y = self.hourglasses[i](x)
            y = self.first_lin[i](y)
            y = self.second_lin[i](y)
            next_skip = y.clone()
            score_map = self.to_score_map[i](y)
            maps.append(score_map)
            if i < self.n_stacks - 1:
                z = self.from_score_map[i](score_map)
                a = torch.cat((y, skip), dim=1)
                a = self.skip_convs[i](a)
                x = z + a
            skip = next_skip
        return maps


class Lin(nn.Module):
    """
    "Lin" layer as implemented in: https://github.com/umich-vl/pose-hg-demo/blob/master/stacked-hourglass-model.lua
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor.
        :param in_channels: int, input channels.
        :param out_channels: int, desired output channels.
        """
        super(Lin, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=True), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True))

    def forward(self, x):
        return self.model(x)


class LOSHourGlassBlock(nn.Module):
    """
    HourGlassBlock, repeated in an hourglass-type network.
    """
    EXPANSION = 2

    def __init__(self, graph, in_channels, out_channels, operating_channels=64
        ):
        """
        Constructor.
        :param graph: decoder.LOSComputationGraph, represents the computation flow.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        """
        super(LOSHourGlassBlock, self).__init__()
        self.operating_channels = operating_channels
        self.graph = graph
        samplers = []
        nodes, _ = zip(*self.graph.items())
        nodes = [None] + list(nodes) + [None]
        for i in range(len(nodes[:-1])):
            samplers.append(self.make_sampling(nodes[i], nodes[i + 1]))
        self.samplers = nn.ModuleList(samplers)
        skip_ops = []
        for node in graph.keys():
            if node.residual:
                skip_ops.append(HourGlassResidual(self.operating_channels,
                    self.operating_channels))
            else:
                skip_ops.append(None)
        last_node = list(graph.keys())[-1]
        res = last_node.residual_node
        if res:
            skip_ops[res.idx] = HourGlassResidual(self.operating_channels,
                out_channels)
        self.skip_ops = nn.ModuleList(skip_ops)
        path_ops = [HourGlassResidual(in_channels, self.operating_channels)]
        for i in range(len(graph) - 2):
            path_ops.append(HourGlassResidual(self.operating_channels, self
                .operating_channels))
        path_ops.append(HourGlassResidual(self.operating_channels,
            out_channels))
        self.path_ops = nn.ModuleList(path_ops)

    @staticmethod
    def make_sampling(prev_node, next_node):
        """
        Determine the factor of up/down sampling needed to move between two nodes.
        :param prev_node: LOSComputationGraph.Node | None.
        :param next_node: LOSComputationGraph.Node.
        :return: nn.MaxPool2d | nn.Upsample
        """
        if prev_node is None:
            prev_node = LOSComputationGraph.Node(1, -1)
        if next_node is None:
            next_node = LOSComputationGraph.Node(1, -1)
        if prev_node.resolution == next_node.resolution:
            return Identity()
        elif prev_node.resolution > next_node.resolution:
            s = int(prev_node.resolution / next_node.resolution)
            return nn.MaxPool2d(kernel_size=2, stride=s)
        else:
            f = int(next_node.resolution / prev_node.resolution)
            return nn.Upsample(scale_factor=f, mode='nearest')

    def forward(self, x):
        residuals = [None for _ in range(len(self.graph))]
        for i, (node, _) in enumerate(self.graph.items()):
            x = self.samplers[i](x)
            x = self.path_ops[i](x)
            if node.residual:
                residuals[i] = self.skip_ops[i](x.clone())
            res = node.residual_node
            if res:
                x += residuals[res.idx]
                residuals[res.idx] = None
        return self.samplers[-1](x)


class HourGlassResidual(nn.Module):
    """
    Hour glass residual, As defined in https://arxiv.org/pdf/1603.06937.pdf.
    Code converted from original lua: https://github.com/umich-vl/pose-hg-demo/blob/master/residual.lua
    """

    def __init__(self, in_channels, out_channels):
        super(HourGlassResidual, self).__init__()
        self.skip_layer = Identity(
            ) if in_channels == out_channels else nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=True), nn.
            BatchNorm2d(out_channels))
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2,
            kernel_size=1, bias=True), nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True), nn.Conv2d(out_channels // 2, 
            out_channels // 2, kernel_size=3, stride=1, padding=1, bias=
            True), nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=
            True), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        Apply forward propagation operation.
        :param x: Variable, input.
        :return: Variable
        """
        residual = x
        out = self.model(x)
        return out + self.skip_layer(residual)


class ResidualPhase(nn.Module):
    """
    Residual Genome phase.
    """

    def __init__(self, gene, in_channels, out_channels, idx, preact=False):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        :param preact: should we use the preactivation scheme?
        """
        super(ResidualPhase, self).__init__()
        self.channel_flag = in_channels != out_channels
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)
        if preact:
            node_constructor = PreactResidualNode
        else:
            node_constructor = ResidualNode
        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                nodes.append(node_constructor(out_channels, out_channels))
            else:
                nodes.append(None)
        self.nodes = nn.ModuleList(nodes)
        conv1x1s = [Identity()] + [Identity() for _ in range(max(self.
            dependency_graph.keys()))]
        for node_idx, dependencies in self.dependency_graph.items():
            if len(dependencies) > 1:
                conv1x1s[node_idx] = nn.Conv2d(len(dependencies) *
                    out_channels, out_channels, kernel_size=1, bias=False)
        self.processors = nn.ModuleList(conv1x1s)
        self.out = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True))

    @staticmethod
    def build_dependency_graph(gene):
        """
        Build a graph describing the connections of a phase.
        "Repairs" made are as follows:
            - If a node has no input, but gives output, connect it to the input node (index 0 in outputs).
            - If a node has input, but no output, connect it to the output node (value returned from forward method).
        :param gene: gene describing the phase connections.
        :return: dict
        """
        graph = {}
        residual = gene[-1][0] == 1
        graph[1] = []
        for i in range(len(gene) - 1):
            graph[i + 2] = [(j + 1) for j in range(len(gene[i])) if gene[i]
                [j] == 1]
        graph[len(gene) + 1] = [0] if residual else []
        no_inputs = []
        no_outputs = []
        for i in range(1, len(gene) + 1):
            if len(graph[i]) == 0:
                no_inputs.append(i)
            has_output = False
            for j in range(i + 1, len(gene) + 2):
                if i in graph[j]:
                    has_output = True
                    break
            if not has_output:
                no_outputs.append(i)
        for node in no_outputs:
            if node not in no_inputs:
                graph[len(gene) + 1].append(node)
        for node in no_inputs:
            if node not in no_outputs:
                graph[node].append(0)
        return graph

    def forward(self, x):
        if self.channel_flag:
            x = self.first_conv(x)
        outputs = [x]
        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:
                outputs.append(None)
            else:
                outputs.append(self.nodes[i - 1](self.process_dependencies(
                    i, outputs)))
        return self.out(self.process_dependencies(len(self.nodes) + 1, outputs)
            )

    def process_dependencies(self, node_idx, outputs):
        """
        Process dependencies with a depth-wise concatenation and
        :param node_idx: int,
        :param outputs: list, current outputs
        :return: Variable
        """
        return self.processors[node_idx](torch.cat([outputs[i] for i in
            self.dependency_graph[node_idx]], dim=1))


class ResidualNode(nn.Module):
    """
    Basic computation unit.
    Does convolution, batchnorm, and relu (in this order).
    """

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
        padding=1, bias=False):
        """
        Constructor.
        Default arguments preserve dimensionality of input.

        :param in_channels: input to the node.
        :param out_channels: output channels from the node.
        :param stride: stride of convolution, default 1.
        :param kernel_size: size of convolution kernel, default 3.
        :param padding: amount of zero padding, default 1.
        :param bias: true to use bias, false to not.
        """
        super(ResidualNode, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=
            bias), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Apply forward propagation operation.
        :param x: Variable, input.
        :return: Variable.
        """
        return self.model(x)


class PreactResidualNode(nn.Module):
    """
    Basic computation unit.
    Does batchnorm, relu, and convolution (in this order).
    """

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
        padding=1, bias=False):
        """
        Constructor.
        Default arguments preserve dimensionality of input.

        :param in_channels: input to the node.
        :param out_channels: output channels from the node.
        :param stride: stride of convolution, default 1.
        :param kernel_size: size of convolution kernel, default 3.
        :param padding: amount of zero padding, default 1.
        :param bias: true to use bias, false to not.
        """
        super(PreactResidualNode, self).__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size
            =kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        """
        Apply forward propagation operation.
        :param x: Variable, input.
        :return: Variable.
        """
        return self.model(x)


class DensePhase(nn.Module):
    """
    Phase with nodes that operates like DenseNet's bottle necking and growth rate scheme.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(self, gene, in_channels, out_channels, idx):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        """
        super(DensePhase, self).__init__()
        self.in_channel_flag = in_channels != out_channels
        self.out_channel_flag = out_channels != DenseNode.t
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)
        channel_adjustment = 0
        for dep in self.dependency_graph[len(gene) + 1]:
            if dep == 0:
                channel_adjustment += out_channels
            else:
                channel_adjustment += DenseNode.t
        self.last_conv = nn.Conv2d(channel_adjustment, out_channels,
            kernel_size=1, stride=1, bias=False)
        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                channels = self.compute_channels(self.dependency_graph[i + 
                    1], out_channels)
                nodes.append(DenseNode(channels))
            else:
                nodes.append(None)
        self.nodes = nn.ModuleList(nodes)
        self.out = nn.Sequential(self.last_conv, nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True))

    @staticmethod
    def compute_channels(dependency, out_channels):
        """
        Compute the number of channels incoming to a node.
        :param dependency: list, nodes that a particular node gets input from.
        :param out_channels: int, desired number of output channels from the phase.
        :return: int
        """
        channels = 0
        for d in dependency:
            if d == 0:
                channels += out_channels
            else:
                channels += DenseNode.t
        return channels

    def forward(self, x):
        if self.in_channel_flag:
            x = self.first_conv(x)
        outputs = [x]
        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:
                outputs.append(None)
            else:
                outputs.append(self.nodes[i - 1](torch.cat([outputs[j] for
                    j in self.dependency_graph[i]], dim=1)))
        if self.out_channel_flag and 0 in self.dependency_graph[len(self.
            nodes) + 1]:
            non_zero_dep = [dep for dep in self.dependency_graph[len(self.
                nodes) + 1] if dep != 0]
            return self.out(torch.cat([outputs[i] for i in non_zero_dep] +
                [outputs[0]], dim=1))
        if self.out_channel_flag:
            return self.out(torch.cat([outputs[i] for i in self.
                dependency_graph[len(self.nodes) + 1]], dim=1))
        return self.out(torch.cat([outputs[i] for i in self.
            dependency_graph[len(self.nodes) + 1]]))


class DenseNode(nn.Module):
    """
    Node that operates like DenseNet layers.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """
    t = 64
    k = 4

    def __init__(self, in_channels):
        """
        Constructor.
        Only needs number of input channels, everything else is automatic from growth rate and DenseNet specs.
        :param in_channels: int, input channels.
        """
        super(DenseNode, self).__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels, self.t * self.k,
            kernel_size=1, bias=False), nn.BatchNorm2d(self.t * self.k), nn
            .ReLU(inplace=True), nn.Conv2d(self.t * self.k, self.t,
            kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.model(x)


class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def phase_active(gene):
    """
    Determine if a phase is active.
    :param gene: list, gene describing a phase.
    :return: bool, true if active.
    """
    return sum([sum(t) for t in gene[:-1]]) != 0


class ChannelBasedDecoder(Decoder):
    """
    Channel based decoder that deals with encapsulating constructor logic.
    """

    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome)
        self._model = None
        self._genome = self.get_effective_genome(list_genome)
        self._channels = channels[:len(self._genome)]
        if repeats is not None:
            active_repeats = []
            for idx, gene in enumerate(list_genome):
                if phase_active(gene):
                    active_repeats.append(repeats[idx])
            self.adjust_for_repeats(active_repeats)
        else:
            self._repeats = [(1) for _ in self._genome]
        if not self._genome:
            self._model = Identity()

    def adjust_for_repeats(self, repeats):
        """
        Adjust for repetition of phases.
        :param repeats:
        """
        self._repeats = repeats
        repeated_genome = []
        repeated_channels = []
        for i, repeat in enumerate(self._repeats):
            for j in range(repeat):
                if j == 0:
                    repeated_channels.append((self._channels[i][0], self.
                        _channels[i][1]))
                else:
                    repeated_channels.append((self._channels[i][1], self.
                        _channels[i][1]))
                repeated_genome.append(self._genome[i])
        self._genome = repeated_genome
        self._channels = repeated_channels

    def build_layers(self, phases):
        """
        Build up the layers with transitions.
        :param phases: list of phases
        :return: list of layers (the model).
        """
        layers = []
        last_phase = phases.pop()
        for phase, repeat in zip(phases, self._repeats):
            for _ in range(repeat):
                layers.append(phase)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(last_phase)
        return layers

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


class DenseGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """

    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome, channels, repeats=repeats)
        if self._model is not None:
            return
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self.
            _genome, self._channels)):
            phases.append(DensePhase(gene, in_channels, out_channels, idx))
        self._model = nn.Sequential(*self.build_layers(phases))

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class ResidualGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """

    def __init__(self, list_genome, channels, preact=False, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome, channels, repeats=repeats)
        if self._model is not None:
            return
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self.
            _genome, self._channels)):
            phases.append(ResidualPhase(gene, in_channels, out_channels,
                idx, preact=preact))
        self._model = nn.Sequential(*self.build_layers(phases))

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class VariableGenomeDecoder(ChannelBasedDecoder):
    """
    Residual decoding with extra integer for type of node inside the phase.
    This genome decoder produces networks that are a superset of ResidualGenomeDecoder networks.
    """
    RESIDUAL = 0
    PREACT_RESIDUAL = 1
    DENSE = 2

    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network, and the type of phase.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        phase_types = [gene.pop() for gene in list_genome]
        genome_copy = copy(list_genome)
        super().__init__(list_genome, channels, repeats=repeats)
        if self._model is not None:
            return
        self._types = self.adjust_types(genome_copy, phase_types)
        phases = []
        for idx, (gene, (in_channels, out_channels), phase_type) in enumerate(
            zip(self._genome, self._channels, self._types)):
            if phase_type == self.RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels,
                    idx))
            elif phase_type == self.PREACT_RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels,
                    idx, preact=True))
            elif phase_type == self.DENSE:
                phases.append(DensePhase(gene, in_channels, out_channels, idx))
            else:
                raise NotImplementedError(
                    'Phase type corresponding to {} not implemented.'.
                    format(phase_type))
        self._model = nn.Sequential(*self.build_layers(phases))

    def adjust_types(self, genome, phase_types):
        """
        Get only the phases that are active.
        Similar to ResidualDecoder.get_effective_genome but we need to consider phases too.
        :param genome: list, list of ints
        :param phase_types: list,
        :return:
        """
        effective_types = []
        for idx, (gene, phase_type) in enumerate(zip(genome, phase_types)):
            if phase_active(gene):
                for _ in range(self._repeats[idx]):
                    effective_types.append(*phase_type)
        return effective_types

    def get_model(self):
        return self._model


def get_decoder(decoder_str, genome, channels, repeats=None):
    """
    Construct the appropriate decoder.
    :param decoder_str: string, refers to what genome scheme we're using.
    :param genome: list, list of genomes.
    :param channels: list, list of channel sizes.
    :param repeats: None | list, how many times to repeat each phase.
    :return: evolution.ChannelBasedDecoder
    """
    if decoder_str == 'residual':
        return ResidualGenomeDecoder(genome, channels, repeats=repeats)
    if decoder_str == 'swapped-residual':
        return ResidualGenomeDecoder(genome, channels, preact=True, repeats
            =repeats)
    if decoder_str == 'dense':
        return DenseGenomeDecoder(genome, channels, repeats=repeats)
    if decoder_str == 'variable':
        return VariableGenomeDecoder(genome, channels, repeats=repeats)
    raise NotImplementedError('Decoder {} not implemented.'.format(decoder_str)
        )


class EvoNetwork(nn.Module):
    """
    Entire network.
    Made up of Phases.
    """

    def __init__(self, genome, channels, out_features, data_shape, decoder=
        'residual', repeats=None):
        """
        Network constructor.
        :param genome: depends on decoder scheme, for most this is a list.
        :param channels: list of desired channel tuples.
        :param out_features: number of output features.
        :param decoder: string, what kind of decoding scheme to use.
        """
        super(EvoNetwork, self).__init__()
        assert len(channels) == len(genome
            ), 'Need to supply as many channel tuples as genes.'
        if repeats is not None:
            assert len(repeats) == len(genome
                ), 'Need to supply repetition information for each phase.'
        self.model = get_decoder(decoder, genome, channels, repeats).get_model(
            )
        out = self.model(torch.autograd.Variable(torch.zeros(1, channels[0]
            [0], *data_shape)))
        shape = out.data.shape
        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)
        shape = self.gap(out).data.shape
        self.linear = nn.Linear(shape[1] * shape[2] * shape[3], out_features)
        self.model.zero_grad()

    def forward(self, x):
        """
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        """
        x = self.gap(self.model(x))
        x = x.view(x.size(0), -1)
        return self.linear(x), None


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda
    C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
    count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine: nn.
    MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C,
    stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C,
    affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C,
    3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine:
    SepConv(C, C, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C,
    stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2,
    affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C,
    5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C, stride,
    affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 7),
    stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C, C, (7, 1),
    stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C,
    affine=affine))}


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).
            bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
        reduction_prev, SE=False):
        super(Cell, self).__init__()
        None
        self.se_layer = None
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return self.se_layer(torch.cat([states[i] for i in self._concat
                ], dim=1))


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(
            inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=
            False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                reduction_prev, SE=SE)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary,
                num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class PyramidNetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype,
        increment=4, SE=False):
        super(PyramidNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=
            False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                reduction_prev, SE=SE)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
            C_curr += increment
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary,
                num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.
            ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1,
            bias=False), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary,
                num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in,
            C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(
            C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1,
            padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn
            .ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=
            kernel_size, stride=1, padding=padding, groups=C_in, bias=False
            ), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction,
            bias=False), nn.ReLU(inplace=True), nn.Linear(channel //
            reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ianwhale_nsga_net(_paritybench_base):
    pass
    def test_000(self):
        self._check(DenseNode(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DilConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(FactorizedReduce(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(HourGlassResidual(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(LOSHourGlassDecoder(*[], **{'genome': [4, 4], 'n_stacks': 4, 'out_feature_maps': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_006(self):
        self._check(Lin(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(PreactResidualNode(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ReLUConvBN(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(ResidualNode(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(SepConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(Zero(*[], **{'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

