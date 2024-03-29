import sys
_module = sys.modules[__name__]
del sys
tests = _module
test_meta_comp_graph = _module
torchmtl = _module
model_builder = _module
wrapping_layers = _module

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


import logging


import torch


from torch import nn


from torch import optim


from torch.nn import Sequential


from torch.nn import Linear


from torch.nn import MSELoss


from copy import copy


ANCHOR_KEY = 'anchor_layer'


AUTO_WEIGHT_KEY = 'auto'


LAYER_KEY = 'layers'


LOSS_KEY = 'loss'


LOSS_REG_KEY = 'loss_weight'


MISSING_WEIGHT_MSG = 'Expect {0} for task {1} but none provided.'


NAME_KEY = 'name'


WEIGHT_INIT_KEY = 'loss_init_val'


class MTLModel(nn.Module):
    """
    A torch.nn.Module built from a set of shared and task specific layers

    Attributes
    ----------
    g : networkx.Graph
        The meta-computation graph

    task_layers : list
        A list which holds the layers for which to build the computation graph

    output_tasks : list
        A list which holds the tasks for which the output should be returned

    layer_names : list
        A list of the names of each layer

    losses : dict
        A dictionary which maps the name of a layer to its loss function

    loss_weights : dict
        A dictionary which maps the name of a layer to the weight of its loss
        function
    """

    def __init__(self, task_layers, output_tasks):
        super(MTLModel, self).__init__()
        self.task_layers = task_layers
        self.output_tasks = output_tasks
        self.layer_names = [t[NAME_KEY] for t in task_layers]
        self._initialize_graph()
        self._initialize_losses()
        self._initialize_loss_weights()

    def _initialize_losses(self):
        self.losses = {task[NAME_KEY]: task[LOSS_KEY] for task in self.task_layers if LOSS_KEY in task.keys()}

    def _initialize_loss_weights(self):
        self.loss_weights = {}
        for task in self.task_layers:
            self._set_loss_weight(task)

    def _set_loss_weight(self, task):
        task_name = task[NAME_KEY]
        if LOSS_REG_KEY in task.keys():
            if task[LOSS_REG_KEY] == AUTO_WEIGHT_KEY:
                assert WEIGHT_INIT_KEY in task.keys(), MISSING_WEIGHT_MSG.format(WEIGHT_INIT_KEY, task_name)
                loss_weight = task[WEIGHT_INIT_KEY]
                loss_name = f'{task_name}_loss'
                loss_weight = torch.nn.Parameter(torch.full((1,), loss_weight))
                setattr(self, loss_name, loss_weight)
                self.loss_weights[task_name] = getattr(self, loss_name)
            else:
                self.loss_weights[task_name] = task[LOSS_REG_KEY]

    def _initialize_graph(self):
        self.g = nx.DiGraph()
        self.g.add_node('root')
        self._build_graph()

    def _bfs_forward(self, start_node):
        """ Here we iteratore through the graph in a BFS-fashion starting from
        `start_node`, typically this is the `root` node. This node is skipped
        and we pass the input data and resulting outputs from all layers foward.
        """
        visited = {node: (False) for node in self.layer_names}
        queue = [start_node]
        visited[start_node] = True
        while queue:
            node = queue.pop(0)
            if node != start_node:
                input_nodes = self.g.predecessors(node)
                if logging.getLogger().level == logging.DEBUG:
                    l = copy(input_nodes)
                    None
                cur_layer = getattr(self, node)
                output_pre_layers = []
                output_complete = True
                for n in input_nodes:
                    if n not in self.outputs.keys():
                        if logging.getLogger().level == logging.DEBUG:
                            None
                        output_complete = False
                        break
                    else:
                        output_pre_layers.append(self.outputs[n])
                if not output_complete:
                    if logging.getLogger().level == logging.DEBUG:
                        None
                    queue.append(node)
                else:
                    cur_output = cur_layer(*output_pre_layers)
                    self.outputs[node] = cur_output
            for i in self.g.successors(node):
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        losses, loss_weights = self._get_losses()
        return [self.outputs[t] for t in self.output_tasks], losses, loss_weights

    def _get_losses(self):
        losses = []
        loss_weights = []
        for t in self.output_tasks:
            losses.append(self.losses.get(t))
            loss_weights.append(self.loss_weights.get(t))
        return losses, loss_weights

    def _build_graph(self):
        for layer in self.task_layers:
            self._add_layer(layer)
            self._add_to_graph(layer)

    def _add_to_graph(self, layer):
        layer_name = layer[NAME_KEY]
        self._add_node(layer_name)
        if 'anchor_layer' not in layer.keys():
            self.g.add_edge('root', layer_name)
        else:
            anchor_layer = layer[ANCHOR_KEY]
            if isinstance(anchor_layer, list):
                for a_l_name in anchor_layer:
                    self._add_node(a_l_name)
                    self.g.add_edge(a_l_name, layer_name)
            else:
                self._add_node(anchor_layer)
                self.g.add_edge(anchor_layer, layer_name)

    def _add_node(self, layer):
        if isinstance(layer, str):
            layer_name = layer
            self.g.add_node(layer_name)
        else:
            layer_name = layer[NAME_KEY]
            self.g.add_node(layer_name)
            if 'anchor_layer' not in layer.keys():
                self.g.add_edge('root', layer_name)

    def _add_layer(self, layer):
        layer_modules = layer[LAYER_KEY]
        layer_name_main = layer[NAME_KEY]
        setattr(self, layer_name_main, layer_modules)

    def forward(self, input):
        self.outputs = {'root': input}
        return self._bfs_forward('root')

