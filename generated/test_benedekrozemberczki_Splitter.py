import sys
_module = sys.modules[__name__]
del sys
ego_splitting = _module
main = _module
param_parser = _module
splitter = _module
utils = _module
walkers = _module

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


import torch


import random


import numpy as np


class Splitter(torch.nn.Module):
    """
    An implementation of "Splitter: Learning Node Representations
    that Capture Multiple Social Contexts" (WWW 2019).
    Paper: http://epasto.org/papers/www2019splitter.pdf
    """

    def __init__(self, args, base_node_count, node_count):
        """
        Splitter set up.
        :param args: Arguments object.
        :param base_node_count: Number of nodes in the source graph.
        :param node_count: Number of nodes in the persona graph.
        """
        super(Splitter, self).__init__()
        self.args = args
        self.base_node_count = base_node_count
        self.node_count = node_count

    def create_weights(self):
        """
        Creating weights for embedding.
        """
        self.base_node_embedding = torch.nn.Embedding(self.base_node_count,
            self.args.dimensions, padding_idx=0)
        self.node_embedding = torch.nn.Embedding(self.node_count, self.args
            .dimensions, padding_idx=0)
        self.node_noise_embedding = torch.nn.Embedding(self.node_count,
            self.args.dimensions, padding_idx=0)

    def initialize_weights(self, base_node_embedding, mapping):
        """
        Using the base embedding and the persona mapping for initializing the embeddings.
        :param base_node_embedding: Node embedding of the source graph.
        :param mapping: Mapping of personas to nodes.
        """
        persona_embedding = np.array([base_node_embedding[n] for _, n in
            mapping.items()])
        self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(
            persona_embedding))
        self.node_noise_embedding.weight.data = torch.nn.Parameter(torch.
            Tensor(persona_embedding))
        self.base_node_embedding.weight.data = torch.nn.Parameter(torch.
            Tensor(base_node_embedding), requires_grad=False)

    def calculate_main_loss(self, sources, contexts, targets):
        """
        Calculating the main embedding loss.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :return main_loss: Loss value.
        """
        node_f = self.node_embedding(sources)
        node_f = torch.nn.functional.normalize(node_f, p=2, dim=1)
        feature_f = self.node_noise_embedding(contexts)
        feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1)
        scores = torch.sum(node_f * feature_f, dim=1)
        scores = torch.sigmoid(scores)
        main_loss = targets * torch.log(scores) + (1 - targets) * torch.log(
            1 - scores)
        main_loss = -torch.mean(main_loss)
        return main_loss

    def calculate_regularization(self, pure_sources, personas):
        """
        Calculating the regularization loss.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return regularization_loss: Loss value.
        """
        source_f = self.node_embedding(pure_sources)
        original_f = self.base_node_embedding(personas)
        scores = torch.clamp(torch.sum(source_f * original_f, dim=1), -15, 15)
        scores = torch.sigmoid(scores)
        regularization_loss = -torch.mean(torch.log(scores))
        return regularization_loss

    def forward(self, sources, contexts, targets, personas, pure_sources):
        """
        Doing a forward pass.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return loss: Loss value.
        """
        main_loss = self.calculate_main_loss(sources, contexts, targets)
        regularization_loss = self.calculate_regularization(pure_sources,
            personas)
        loss = main_loss + self.args.lambd * regularization_loss
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_benedekrozemberczki_Splitter(_paritybench_base):
    pass
