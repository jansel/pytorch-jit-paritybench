import sys
_module = sys.modules[__name__]
del sys
master = _module
model = _module
train = _module
utils = _module

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


import torch.nn as nn


import torch


import math


import torch.distributed as dist


import torch.backends.cudnn as cudnn


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader


class depthwise_separable_conv_3x3(nn.Module):

    def __init__(self, nin, nout, stride):
        super(depthwise_separable_conv_3x3, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride,
            padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Triplet_unit(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = depthwise_separable_conv_3x3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


class Node_OP(nn.Module):

    def __init__(self, Node, inplanes, outplanes):
        super(Node_OP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(Node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        if self.is_input_node:
            self.conv = Triplet_unit(inplanes, outplanes, stride=2)
        else:
            self.conv = Triplet_unit(outplanes, outplanes, stride=1)

    def forward(self, *input):
        if self.input_nums > 1:
            out = self.sigmoid(self.mean_weight[0]) * input[0]
            for i in range(1, self.input_nums):
                out = out + self.sigmoid(self.mean_weight[i]) * input[i]
        else:
            out = input[0]
        out = self.conv(out)
        return out


def load_graph(path):
    return nx.read_yaml(path)


def save_graph(graph, path):
    nx.write_yaml(graph, path)


def build_graph(Nodes, args):
    if args.graph_model == 'ER':
        return nx.random_graphs.erdos_renyi_graph(Nodes, args.P, args.seed)
    elif args.graph_model == 'BA':
        return nx.random_graphs.barabasi_albert_graph(Nodes, args.M, args.seed)
    elif args.graph_model == 'WS':
        return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.
            K, args.P, tries=200, seed=args.seed)


class CNN(nn.Module):

    def __init__(self, args, num_classes=1000):
        super(CNN, self).__init__()
        self.conv1 = depthwise_separable_conv_3x3(3, args.channels // 2, 2)
        self.bn1 = nn.BatchNorm2d(args.channels // 2)
        if args.net_type == 'small':
            self.conv2 = Triplet_unit(args.channels // 2, args.channels, 2)
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv3.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv3.yaml'))
            self.conv3 = StageBlock(graph, args.channels, args.channels)
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv4.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv4.yaml'))
            self.conv4 = StageBlock(graph, args.channels, args.channels * 2)
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv5.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv5.yaml'))
            self.conv5 = StageBlock(graph, args.channels * 2, args.channels * 4
                )
            self.relu = nn.ReLU()
            self.conv = nn.Conv2d(args.channels * 4, 1280, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(1280)
        elif args.net_type == 'regular':
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv2.yaml'))
            else:
                graph = build_graph(args.nodes // 2, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv2.yaml'))
            self.conv2 = StageBlock(graph, args.channels // 2, args.channels)
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv3.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv3.yaml'))
            self.conv3 = StageBlock(graph, args.channels, args.channels * 2)
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv4.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv4.yaml'))
            self.conv4 = StageBlock(graph, args.channels * 2, args.channels * 4
                )
            if args.resume:
                graph = load_graph(os.path.join(args.model_dir, 'conv5.yaml'))
            else:
                graph = build_graph(args.nodes, args)
                save_graph(graph, os.path.join(args.model_dir, 'conv5.yaml'))
            self.conv5 = StageBlock(graph, args.channels * 4, args.channels * 8
                )
            self.relu = nn.ReLU()
            self.conv = nn.Conv2d(args.channels * 8, 1280, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1280, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_JiaminRen_RandWireNN(_paritybench_base):
    pass

    def test_000(self):
        self._check(depthwise_separable_conv_3x3(*[], **{'nin': 4, 'nout': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Triplet_unit(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 4, 4, 4])], {})
