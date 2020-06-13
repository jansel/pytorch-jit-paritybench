import sys
_module = sys.modules[__name__]
del sys
capsule_network = _module
capsule_network_svhn = _module

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


import torch.nn.functional as F


from torch import nn


import numpy as np


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1,
        transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, 
        len(input.size()) - 1)


NUM_ROUTING_ITERATIONS = 3


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels,
        out_channels, kernel_size=None, stride=None, num_iterations=
        NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules,
                num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Conv2d(in_channels,
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=0) for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[(None), :, :, (None), :] @ self.route_weights[:, (
                None), :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)
                    )
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in
                self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


NUM_CLASSES = 10


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size
            =9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8,
            num_route_nodes=-1, in_channels=256, out_channels=32,
            kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES,
            num_route_nodes=32 * 6 * 6, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(nn.Linear(16 * NUM_CLASSES, 512), nn.
            ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=
                max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, (None)]).view(x.size(0),
            -1))
        return classes, reconstructions


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels,
        out_channels, kernel_size=None, stride=None, num_iterations=
        NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules,
                num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Conv2d(in_channels,
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=0) for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[(None), :, :, (None), :] @ self.route_weights[:, (
                None), :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)
                    )
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in
                self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size
            =9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8,
            num_route_nodes=-1, in_channels=256, out_channels=32,
            kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES,
            num_route_nodes=2048, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(nn.Linear(16 * NUM_CLASSES, 512), nn.
            ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 3072), nn.Sigmoid())

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=
                max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, (None)]).view(x.size(0),
            -1))
        return classes, reconstructions


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gram_ai_capsule_networks(_paritybench_base):
    pass
    def test_000(self):
        self._check(CapsuleLoss(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

