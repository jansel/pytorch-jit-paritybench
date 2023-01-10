import sys
_module = sys.modules[__name__]
del sys
src = _module
applications = _module
measures = _module
feature_similarity = _module
inter_stress = _module
stress_points = _module
trainers = _module
averaging = _module
base_trainer = _module
gradnorm = _module
mgda = _module
naive = _module
unzipping = _module
datasets = _module
toy = _module
models = _module
hydra_base = _module
lenet = _module
resnet = _module
run = _module
utils = _module
config_utils = _module
grad_normalizers = _module
graph_clustering = _module
log_utils = _module
losses = _module
metrics = _module
min_norm_solver = _module
regularizers = _module

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


import torch


import torch.optim as optim


import numpy as np


import pandas as pd


import torch.nn as nn


from copy import deepcopy


from collections import deque


from collections import OrderedDict


import torch.nn.functional as F


class BatchNormPillow(nn.Module):
    """
    Customized Batch Normalization, for which we can access the inner
    representation (pre-affine).

    Attributes:
      raw_bn:       an instance of `nn.BatchNorm_`, with `affine=False`
      weight, bias: gamma and beta coefficients (learnable)
      rep:          inner representation (saved if retain_rep is True)
      retain_rep:   whether to retain the result of raw_bn
    """

    def __init__(self, channels, bn_type='2d'):
        super().__init__()
        if bn_type == '1d':
            self.raw_bn = nn.BatchNorm1d(channels, affine=False)
        elif bn_type == '2d':
            self.raw_bn = nn.BatchNorm2d(channels, affine=False)
        else:
            raise RuntimeError('Only 1D and 2D BN-Pillow are supported')
        self.weight = nn.Parameter(torch.empty((channels,)).uniform_())
        self.bias = nn.Parameter(torch.zeros((channels,)))
        self.rep = None
        self.retain_rep = False

    def forward(self, x):
        x = self.raw_bn(x)
        if self.retain_rep:
            self.rep = x
        y = torch.transpose(x, 1, -1) * self.weight + self.bias
        return torch.transpose(y, 1, -1)


class Block(nn.Module):
    """
    A wrapper around `nn.Module` that holds convenient parameters for the
    Hydra class, which otherwise would be hard to access or require.

    Attributes:
      module:           an `nn.Module` that we will wrap this around
      with_bn_pillow:   whether to put a batch-normalization layer after
      bn_pillow:        the batchnorm layer mentioned, created in runtime
      trainable:        DO NOT confuse with nn.Module.training (module state)
    """

    def __init__(self, module, bn_pillow_planes=None, bn_pillow_type='2d'):
        super().__init__()
        self.add_module('module', module)
        self.trainable = True
        if bn_pillow_planes is not None:
            self.with_bn_pillow = True
            bn_pillow = BatchNormPillow(bn_pillow_planes, bn_pillow_type)
            self.add_module('bn_pillow', bn_pillow)
        else:
            self.with_bn_pillow = False

    def forward(self, x, *args, **kwargs):
        y = self.module.forward(x, *args, **kwargs)
        if self.with_bn_pillow:
            if not hasattr(self, 'bn_pillow'):
                pillow_type = '2d' if len(y.shape) == 4 else '1d'
                bn_pillow = BatchNormPillow(y.shape[1], pillow_type)
                device = next(self.module.parameters()).device
                bn_pillow = bn_pillow
                self.add_module('bn_pillow', bn_pillow)
                if self.training:
                    self.bn_pillow.train()
                else:
                    self.bn_pillow.eval()
            return self.bn_pillow.forward(y)
        return y


class Controller:
    """
    Hydra's block controller. Stores information about its index in the
    blocks list, the execution chain (blocks that should be executed in
    order before this block), and the children blocks of this block.

    Attributes:
      index:             the index of this block in the Hydra.blocks
      execution_chain:   indices of blocks to be executed prior to this
      parent_index:      index (in Hydra.blocks) of the parent block
      children_indices:  indices (in Hydra.blocks) of the childrens
      task_id:           if this block is a head, stores the task_id
      serving_tasks:     a dict {task_id: idk_what_this_is}
    """

    def __init__(self, index=None):
        self.index = index
        self.execution_chain = [index]
        self.parent_index = None
        self.children_indices = []
        self.task_id = None
        self.serving_tasks = dict()

    def stack_on(self, controller):
        """Stacks current controller on top of another controller"""
        prev_chain = controller.execution_chain.copy()
        self.execution_chain = prev_chain + [self.index]
        self.parent_index = controller.index
        controller.children_indices.append(self.index)
        return self

    def __str__(self):
        return '({}): parent={}, children={}, serving=[{}]'.format(self.index, self.parent_index, self.children_indices, ', '.join(str(task_id) for task_id in self.serving_tasks))

    def __repr__(self):
        return str(self)

    def serialize(self):
        """Serialize to ordinary python's dict object"""
        return self.__dict__

    def deserialize(self, serialized_controller):
        """Deserialize from a python's dict object"""
        for k, v in serialized_controller.items():
            setattr(self, k, v)
        return self


class Hydra(nn.Module):
    """
    A base class for all Multi-Task Neural Networks with hard-shared
    parameters and arbitrary branching schema.

    Attributes:
      blocks:            a `nn.ModuleList` of building blocks of Hydra
      controllers:       a list of controllers accompanying each block
      heads:             dictionary {task_id: index} of Hydra's heads
      rep_tensors:       stores the tensors at branching points
      branching_points:  indices of blocks with more than one children
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.controllers = list()
        self.heads = dict()
        self.rep_tensors = dict()
        self.branching_points = set()

    def add_block(self, module):
        """
        Registers a new Hydra block, automatically adds it to the
        self.blocks and the execution graph.

        Args:
          module: a `nn.Module` object

        Returns:
          a Controller object for newly added block
        """
        new_index = len(self.blocks)
        new_controller = Controller(new_index)
        self.blocks.append(module)
        self.controllers.append(new_controller)
        return new_controller

    def add_head(self, module, task_id):
        """
        Registers a new Hydra block as a "Head". Same as the method
        `register_block()`, but adds the controller to self.heads.

        Args:
          module:    a `nn.Module` object
          task_id:  an identifier of the task that the head is solving

        Returns:
          a Controller object for newly added block
        """
        new_controller = self.add_block(module)
        new_controller.task_id = task_id
        self.heads[task_id] = new_controller.index
        return new_controller

    def extra_repr(self):
        """
        To be displayed each time one calls `repr()`, together with
        the default output of `nn.Module`.
        """
        items = '\n  '.join(str(c) for c in self.controllers)
        controllers = '(block controllers):\n  ' + items
        items = '\n  '.join('({}) -> {}'.format(k, str(c)) for k, c in self.heads.items())
        heads = '(heads):\n  ' + items
        return controllers + '\n' + heads

    def execution_plan(self, task_ids):
        """
        Dynamicaly constructs an execution plan, given the identifiers
        of tasks that we want to perform.

        Args:
          task_ids:  an identifier, or list of identifiers of tasks

        Returns:
          execution_order: a list of indices of modules to be executed
          branching_ids:   indices of branching points
        """
        if not isinstance(task_ids, list):
            task_ids = [task_ids]
        execution_order = []
        branching_ids = set()
        for task_id in task_ids:
            branching_point = None
            controller = self.controllers[self.heads[task_id]]
            task_exec_chain = controller.execution_chain
            for i, index in enumerate(task_exec_chain):
                if index not in execution_order:
                    break
                branching_point = index
            execution_order += task_exec_chain[i:].copy()
            if branching_point is not None:
                branching_ids.add(branching_point)
        return execution_order, branching_ids

    def parameters(self, recurse=True, task_ids=None, only_trainable=False):
        """
        Returns an iterator over module parameters. If task_ids
        are specified, returns an iterator only over the parameters
        that affects the outputs on those tasks.

        Args:
          recurse:         whether to yield the parameters of submodules
          task_ids:        whether to yield only task-related parameters
          only_trainable:  whether to yield only trainable parameters

        Yields:
          Parameter: module parameter
        """
        if task_ids is None and not only_trainable:
            for param in super().parameters(recurse):
                yield param
        else:
            if task_ids is None:
                task_ids = list(self.heads.keys())
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                if only_trainable:
                    if not hasattr(self.blocks[index], 'trainable'):
                        continue
                    if self.blocks[index].trainable is not True:
                        continue
                for param in self.blocks[index].parameters():
                    yield param

    def control_blocks(self, task_ids=None):
        """
        Yields an iterator over the blocks. If `task_ids` are specified,
        only blocks flowing towards corresponding heads will be yielded.
        """
        if task_ids is None:
            for controller, block in zip(self.controllers, self.blocks):
                yield controller, block
        else:
            execution_order, _ = self.execution_plan(task_ids)
            for index in execution_order:
                yield self.controllers[index], self.blocks[index]

    def create_branch(self, index, branches, device=None):
        """
        Dynamically clones `self.blocks[index]`, and stacks the branches
        specified by `branches` on top of the newly cloned branch.

        [Before]                         [After]
                    __ ...........           -------O--- ...........
            index  /                        / index
        --O-------O--- branches[0]       --O          __ branches[0]
                   \\__                      \\ clone  /
                       branches[1]           -------O--- branches[1]

        Args:
          index:      index of the block to clone
          branches:   indices of block's children to stach on the clone
          device:     device to spawn the clone on, can be decided later

        Raises:
          ValueError: in case invalid `index` or `branches` are specified

        Returns:
          controller: controller object of the newly created branch
          block:      module of the newly created branch
        """
        if index in self.heads:
            raise ValueError("Cannot split Hydra's head.")
        controller = self.controllers[index]
        for b in branches:
            if b not in controller.children_indices:
                raise ValueError("Indices of branches should be in controller's chilred_indices.")
        are_equal = True
        for b in controller.children_indices:
            if b not in branches:
                are_equal = False
        if are_equal:
            return self.controllers[index], self.blocks[index]
        block = self.blocks[index]
        cloned_block = deepcopy(block)
        if device is not None:
            cloned_block = cloned_block
        cloned_controller = deepcopy(controller)
        new_index = len(self.controllers)
        cloned_controller.index = new_index
        self.blocks.append(cloned_block)
        self.controllers.append(cloned_controller)
        if cloned_controller.parent_index is not None:
            parent = self.controllers[cloned_controller.parent_index]
            parent.children_indices.append(new_index)
        cloned_controller.execution_chain = [(i if i != index else new_index) for i in cloned_controller.execution_chain]
        controller_deque = deque()
        controller_deque.extend(branches)
        while len(controller_deque) > 0:
            tmp_index = controller_deque.popleft()
            tmp_controller = self.controllers[tmp_index]
            if tmp_controller.parent_index == index:
                tmp_controller.parent_index = new_index
            tmp_controller.execution_chain = [(i if i != index else new_index) for i in tmp_controller.execution_chain]
            controller_deque.extend(tmp_controller.children_indices)
        controller.children_indices = [i for i in controller.children_indices if i not in branches]
        cloned_controller.children_indices = branches
        controller.serving_tasks = dict()
        for i in controller.children_indices:
            tmp_controller = self.controllers[i]
            controller.serving_tasks.update(tmp_controller.serving_tasks)
        cloned_controller.serving_tasks = dict()
        for i in cloned_controller.children_indices:
            tmp_controller = self.controllers[i]
            cloned_controller.serving_tasks.update(tmp_controller.serving_tasks)
        self.rep_tensors.clear()
        _, self.branching_points = self.execution_plan(list(self.heads.keys()))
        return cloned_controller, cloned_block

    def split(self, index, branching_scheme, device):
        """
        Splits a Hydra's block into several blocks, according to the
        `branching_scheme`. Results of `split(0, [[1], [2,3], [4,5]])`:

        | B |  (1) (2) (3) (4) (5)     | A |  (1) (2) (3) (4) (5)
        | E |   |   |   |   |   |      | F |   |   |   |   |   |
        | F |   +---+---|---+---+      | T |   |   |---+   |---+
        | O |          (0)             | E |  (0) (6)     (7)
        | R |           |              | R |   |   |       |
        | E |          (*)             |   |  (*)--+-------+

        Args:
          index:            index of the block to split
          branching_scheme: list of list of indices (as example above)
          device:           a device to spawn the new branches on

        Raises:
          ValueError:       in case invalid parameters are specified

        Returns:
          controllers:      list of controllers of splitted branches
          blocks:           list of blocks - the splitted branches
        """
        if index not in self.branching_points:
            raise ValueError('You can only split layers which indices are in `Hydra.branching_points`.')
        controller = self.controllers[index]
        block = self.blocks[index]
        total_branches = set()
        for branch in branching_scheme:
            total_branches.update(set(branch))
        if not total_branches == set(controller.children_indices):
            raise ValueError('The union of the branches should be equal to `controller.children_indices`.')
        for i in range(len(branching_scheme)):
            scheme_a = set(branching_scheme[i])
            for j in range(i + 1, len(branching_scheme)):
                scheme_b = set(branching_scheme[j])
                if not scheme_a.isdisjoint(scheme_b):
                    raise ValueError('The branching schemes should be disjoint to each other.')
        new_controllers, new_blocks = [controller], [block]
        for branch in branching_scheme[1:]:
            tmp_ctrl, tmp_block = self.create_branch(index, branch, device)
            new_controllers.append(tmp_ctrl)
            new_blocks.append(tmp_block)
        return new_controllers, new_blocks

    def rip(self, device):
        """
        Violently rips the model apart. Below are some example results:

        | B |  (x) (y) (z)         | A |  (x) (y) (z)
        | E |   |   |   |          | F |   |   |   |
        | F |   +--(a) (b)         | T |  (a) (e) (b)
        | O |       |   |          | E |   |   |   |
        | R |       +--(c)         | R |   +--(c) (d)
        | E |           |          |   |       |   |
        |   |          (*)         |   |       +--(*)

        Args:
          device: a device to spawn the new branches on, either CPU or GPU

        Returns:
          a dict of lists of tuples {index: [(new_index, children_index)]}
        """
        indices = list(self.branching_points)
        indices.sort(key=lambda i: len(self.controllers[i].execution_chain))
        index_map = dict()
        for index in indices:
            children_indices = self.controllers[index].children_indices
            branching_scheme = [[i] for i in children_indices]
            new_cs, _ = self.split(index, branching_scheme, device)
            index_map[index] = [(c.index, i) for c, i in zip(new_cs, children_indices)]
        return index_map

    def peel(self, task_ids, device=None):
        """
        Peels off a task-specific subnetwork (like a banana). Please note
        that it does NOT copy the paremeters of the `__init__` of your
        network, inherited from Hydra. Results of peel('task_a'):

        | O |  (task_a)   (task_b)        | P |  (task_a)
        | R |      |         |            | E |      |
        | I |      +----+----+            | E |      +----+
        | G |          (0)                | L |          (0)
        | I |           |                 | E |           |
        | N |          (*)                | D |          (*)

        Args:
          task_ids:  `str` or `list` of `str`, related subnets are peeled
          device:    a device to spawn freshly peeled Hydra on

        Returns:
          peeled_hydra: A new Hydra that is only related to secified tasks.
          index_map:    a dict {old_index: new_index} of block correspondence
        """
        execution_order, _ = self.execution_plan(task_ids)
        index_map = dict((idx, i) for i, idx in enumerate(execution_order))
        new_hydra = Hydra()
        for index in execution_order:
            controller = self.controllers[index]
            block = self.blocks[index]
            new_block = deepcopy(block)
            if device is not None:
                new_block = new_block
            if controller.task_id is not None:
                new_hydra.add_head(new_block, controller.task_id)
            else:
                new_hydra.add_block(new_block)
        for index in execution_order:
            new_index = index_map[index]
            controller = self.controllers[index]
            new_controller = new_hydra.controllers[new_index]
            parent_index = controller.parent_index
            if parent_index is not None:
                new_parent_index = index_map[parent_index]
                new_parent = new_hydra.controllers[new_parent_index]
                new_controller.stack_on(new_parent)
        new_hydra.build()
        return new_hydra, index_map

    def build(self):
        """
        Builds the model. Calculates additional stuffs to make the Hydra
        truly powerful.
        """
        for _, head_index in self.heads.items():
            controller = self.controllers[head_index]
            task_id = controller.task_id
            for index in controller.execution_chain:
                idx = len(self.controllers[index].serving_tasks)
                self.controllers[index].serving_tasks[task_id] = idx
        _, self.branching_points = self.execution_plan(list(self.heads.keys()))

    def forward(self, input_tensor, task_ids, retain_tensors=False, retain_all=False):
        """
        Defines the computation performed at every call. Dynamically
        and automatically decides what to run and in what order.

        Args:
          input_tensor:    a common input for specified tasks
          task_ids:        identifiers of tasks to be executed
          retain_tensors:  if True, save branching tensors to rep_tensors
          retain_all:      if True, save ALL tensors at rep_tensors

        Returns:
          A dictionary {task_id: output} of task-specific outputs
        """
        exec_order, branching_ids = self.execution_plan(task_ids)
        x = input_tensor
        outputs = dict()
        for index in exec_order:
            controller = self.controllers[index]
            parent_index = controller.parent_index
            if parent_index not in branching_ids:
                x = self.blocks[index](x)
            else:
                x = self.blocks[index](self.rep_tensors[parent_index])
            if retain_all:
                self.rep_tensors[index] = x
            elif retain_tensors and index in self.branching_points:
                self.rep_tensors[index] = x
            elif index in branching_ids:
                self.rep_tensors[index] = x
            if controller.task_id is not None:
                outputs[controller.task_id] = x
        if isinstance(task_ids, str):
            return outputs[task_ids]
        return outputs

    def serialize(self):
        """Serializes the Hydra into dictionary objects.

        Returns:
          hydra_serial:  a dictionary of Hydra's parameters
          state_dict:    a state dict of `nn.Module` object
        """
        controller_serializations = [c.serialize() for c in self.controllers]
        hydra_serialization = {'controllers': controller_serializations, 'heads': self.heads}
        return hydra_serialization, self.state_dict()

    def deserialize(self, hydra_serialization, state_dict):
        """Reads the Hydra from its serialized representation.

        Args:
          hydra_serial:  a dictionary of Hydra's parameters
          state_dict:    a state dict of `nn.Module` object

        Returns: self
        """
        self.controllers = [Controller().deserialize(c) for c in hydra_serialization['controllers']]
        self.heads = hydra_serialization['heads']
        self.load_state_dict(state_dict)
        return self

    def save(self, basepath):
        """
        Saves the Hydra to disc. The hydra will be saved in two parts:
          * basepath.yaml  -- stores the Hydra's controllers and heads
          * basepath.pth   -- stores the Hydra's weights

        Args:
          basepath: a full path to file (without extension) to save to
        """
        serialized_hydra, state_dict = self.serialize()
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'w') as outfile:
            yaml.dump(serialized_hydra, outfile)
        pth_path = basepath + '.pth'
        torch.save(state_dict, pth_path)

    def load(self, basepath):
        """
        Loads the Hydra from dist. This will try to find two files:
          * basepath.yaml  -- for the Hydra's controllers and heads
          * basepath.pth   -- for the Hydra's weights

        Returns: self
        """
        basepath = os.path.expanduser(basepath)
        yaml_path = basepath + '.yaml'
        with open(yaml_path, 'r') as stream:
            serialized_hydra = yaml.safe_load(stream)
        pth_path = basepath + '.pth'
        state_dict = torch.load(pth_path)
        return self.deserialize(serialized_hydra, state_dict)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class LeHydra(Hydra):
    """An example of a simple LeNet, written in Hydra API
    """

    def __init__(self, heads):
        super().__init__()
        layer1 = Block(nn.Sequential(OrderedDict([('conv', nn.Conv2d(1, 20, 5)), ('relu', nn.ReLU()), ('pool', nn.MaxPool2d(2))])), bn_pillow_planes=20)
        layer2 = Block(nn.Sequential(OrderedDict([('conv', nn.Conv2d(20, 50, 5)), ('relu', nn.ReLU()), ('pool', nn.MaxPool2d(2))])), bn_pillow_planes=50)
        x = self.add_block(layer1)
        x = self.add_block(layer2).stack_on(x)

        def define_head(n_classes):
            return Block(nn.Sequential(OrderedDict([('flatten', Flatten()), ('fc', nn.Linear(4 * 4 * 50, n_classes)), ('softmax', nn.LogSoftmax(dim=1))])))
        for head in heads:
            module = define_head(head['n_classes'])
            h = self.add_head(module, head['task_id'])
            h.stack_on(x)
        self.build()


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(x)
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18(Hydra):
    """
    Pre-Activation ResNet
    https://arxiv.org/abs/1603.05027
    """

    def __init__(self, heads, num_planes=[32, 64, 64, 128], num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 32
        layers = [nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)]
        bn_planes = [self.in_planes]
        layers.extend(self._make_layer(num_planes[0], num_blocks[0], 1))
        layers.extend(self._make_layer(num_planes[1], num_blocks[1], 2))
        layers.extend(self._make_layer(num_planes[2], num_blocks[2], 2))
        layers.extend(self._make_layer(num_planes[3], num_blocks[3], 2))
        bn_planes.extend([num_planes[0]] * num_blocks[0])
        bn_planes.extend([num_planes[1]] * num_blocks[1])
        bn_planes.extend([num_planes[2]] * num_blocks[2])
        bn_planes.extend([num_planes[3]] * num_blocks[3])
        controller = self.add_block(Block(layers[0], bn_pillow_planes=bn_planes[0]))
        for layer, nplanes in zip(layers[1:], bn_planes[1:]):
            new_controller = self.add_block(Block(layer, bn_pillow_planes=nplanes)).stack_on(controller)
            controller = new_controller

        def define_head(n_classes):
            return Block(nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(self.in_planes, n_classes), nn.LogSoftmax(dim=1)]))
        for head in heads:
            module = define_head(head['n_classes'])
            h = self.add_head(module, head['task_id'])
            h.stack_on(controller)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return layers


class MinNormLinearSolver(nn.Module):
    """Solves the min norm problem in case of 2 vectors (lies on a line):
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, v1v1, v1v2, v2v2):
        """Solver execution on scalar products of 2 vectors

        Args:
          v1v1:  scalar product <V1, V1>
          v1v2:  scalar product <V1, V2>
          v2v2:  scalar product <V2, V2>

        Returns:
          gamma: min-norm solution c = (gamma, 1. - gamma)
          cost:  the norm of min-norm point
        """
        if v1v2 >= v1v1:
            return 1.0, v1v1
        if v1v2 >= v2v2:
            return 0.0, v2v2
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-08)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost


class MinNormPlanarSolver(nn.Module):
    """Solves the min norm problem in case the vectors lies on same plane
    """

    def __init__(self, n_tasks):
        super().__init__()
        i_grid = torch.arange(n_tasks)
        j_grid = torch.arange(n_tasks)
        ii_grid, jj_grid = torch.meshgrid(i_grid, j_grid)
        i_triu, j_triu = np.triu_indices(n_tasks, 1)
        self.register_buffer('n', torch.tensor(n_tasks))
        self.register_buffer('i_triu', torch.from_numpy(i_triu))
        self.register_buffer('j_triu', torch.from_numpy(j_triu))
        self.register_buffer('ii_triu', ii_grid[i_triu, j_triu])
        self.register_buffer('jj_triu', jj_grid[i_triu, j_triu])
        self.register_buffer('one', torch.ones(self.ii_triu.shape))
        self.register_buffer('zero', torch.zeros(self.ii_triu.shape))

    @torch.no_grad()
    def line_solver_vectorized(self, v1v1, v1v2, v2v2):
        """Linear case solver, but for collection of vector pairs (Vi, Vj)

        Args:
          v1v1:  vector of scalar product <Vi, Vi>
          v1v2:  vector of scalar product <Vi, Vj>
          v2v2:  vector of scalar product <Vj, Vj>

        Returns:
          gamma: vector of min-norm solution c = (gamma, 1. - gamma)
          cost:  vector of the norm of min-norm point
        """
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-08)
        gamma = gamma.where(v1v2 < v2v2, self.zero)
        gamma = gamma.where(v1v2 < v1v1, self.one)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        cost = cost.where(v1v2 < v2v2, v2v2)
        cost = cost.where(v1v2 < v1v1, v1v1)
        return gamma, cost

    @torch.no_grad()
    def forward(self, grammian):
        """Planar case solver, when Vi lies on the same plane

        Args:
          grammian: grammian matrix G[i, j] = [<Vi, Vj>], G is a nxn tensor

        Returns:
          sol: coefficients c = [c1, ... cn] that solves the min-norm problem
        """
        vivj = grammian[self.ii_triu, self.jj_triu]
        vivi = grammian[self.ii_triu, self.ii_triu]
        vjvj = grammian[self.jj_triu, self.jj_triu]
        gamma, cost = self.line_solver_vectorized(vivi, vivj, vjvj)
        offset = torch.argmin(cost)
        i_min, j_min = self.i_triu[offset], self.j_triu[offset]
        sol = torch.zeros(self.n, device=grammian.device)
        sol[i_min], sol[j_min] = gamma[offset], 1.0 - gamma[offset]
        return sol


class MinNormSolver(nn.Module):
    """Solves the min norm problem in the general case.
    """

    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-06):
        super().__init__()
        self.n = n_tasks
        self.linear_solver = MinNormLinearSolver()
        self.planar_solver = MinNormPlanarSolver(n_tasks)
        n_grid = torch.arange(n_tasks)
        i_grid = torch.arange(n_tasks, dtype=torch.float32) + 1
        ii_grid, jj_grid = torch.meshgrid(n_grid, n_grid)
        self.register_buffer('n_ts', torch.tensor(n_tasks))
        self.register_buffer('i_grid', i_grid)
        self.register_buffer('ii_grid', ii_grid)
        self.register_buffer('jj_grid', jj_grid)
        self.register_buffer('zero', torch.zeros(n_tasks))
        self.register_buffer('stop_crit', torch.tensor(stop_crit))
        self.max_iter = max_iter
        self.two_sol = nn.Parameter(torch.zeros(2))
        self.two_sol.require_grad = False

    @torch.no_grad()
    def projection_to_simplex(self, gamma):
        sorted_gamma, indices = torch.sort(gamma, descending=True)
        tmp_sum = torch.cumsum(sorted_gamma, 0)
        tmp_max = (tmp_sum - 1.0) / self.i_grid
        non_zeros = torch.nonzero(tmp_max[:-1] > sorted_gamma[1:])
        if non_zeros.shape[0] > 0:
            tmax_f = tmp_max[:-1][non_zeros[0][0]]
        else:
            tmax_f = tmp_max[-1]
        return torch.max(gamma - tmax_f, self.zero)

    @torch.no_grad()
    def next_point(self, cur_val, grad):
        proj_grad = grad - torch.sum(grad) / self.n_ts
        lt_zero = torch.nonzero(proj_grad < 0)
        lt_zero = lt_zero.view(lt_zero.numel())
        gt_zero = torch.nonzero(proj_grad > 0)
        gt_zero = gt_zero.view(gt_zero.numel())
        tm1 = -cur_val[lt_zero] / proj_grad[lt_zero]
        tm2 = (1.0 - cur_val[gt_zero]) / proj_grad[gt_zero]
        t = torch.tensor(1.0, device=grad.device)
        tm1_gt_zero = torch.nonzero(tm1 > 1e-07)
        tm1_gt_zero = tm1_gt_zero.view(tm1_gt_zero.numel())
        if tm1_gt_zero.shape[0] > 0:
            t = torch.min(tm1[tm1_gt_zero])
        tm2_gt_zero = torch.nonzero(tm2 > 1e-07)
        tm2_gt_zero = tm2_gt_zero.view(tm2_gt_zero.numel())
        if tm2_gt_zero.shape[0] > 0:
            t = torch.min(t, torch.min(tm2[tm2_gt_zero]))
        next_point = proj_grad * t + cur_val
        next_point = self.projection_to_simplex(next_point)
        return next_point

    @torch.no_grad()
    def forward(self, vecs):
        """General case solver using simplex projection algorithm.

        Args:
          vecs:  2D tensor V, where each row is a vector Vi

        Returns:
          sol: coefficients c = [c1, ... cn] that solves the min-norm problem
        """
        if self.n == 1:
            return vecs[0]
        if self.n == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            self.two_sol[0], cost = self.linear_solver(v1v1, v1v2, v2v2)
            self.two_sol[1] = 1.0 - self.two_sol[0]
            return self.two_sol.clone()
        grammian = torch.mm(vecs, vecs.t())
        sol_vec = self.planar_solver(grammian)
        ii, jj = self.ii_grid, self.jj_grid
        for iter_count in range(self.max_iter):
            grad_dir = -torch.mv(grammian, sol_vec)
            new_point = self.next_point(sol_vec, grad_dir)
            v1v1 = (sol_vec[ii] * sol_vec[jj] * grammian[ii, jj]).sum()
            v1v2 = (sol_vec[ii] * new_point[jj] * grammian[ii, jj]).sum()
            v2v2 = (new_point[ii] * new_point[jj] * grammian[ii, jj]).sum()
            gamma, cost = self.linear_solver(v1v1, v1v2, v2v2)
            new_sol_vec = gamma * sol_vec + (1 - gamma) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < self.stop_crit:
                return sol_vec
            sol_vec = new_sol_vec
        return sol_vec


class MinNormSolverFW(nn.Module):
    """Wrapper over series of algorithms for solving min-norm tasks.
    """

    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-06):
        """Stuffs we don't want to re-define too much times
        """
        super().__init__()
        self.n_tasks = n_tasks
        n = torch.tensor(n_tasks)
        self.MAX_ITER = max_iter
        STOP_CRIT = torch.tensor(stop_crit)
        grammian = torch.empty((n_tasks, n_tasks), dtype=torch.float32)
        sol = torch.empty((n_tasks,), dtype=torch.float32)
        new_sol = torch.empty((n_tasks,), dtype=torch.float32)
        self.register_buffer('n', n)
        self.register_buffer('STOP_CRIT', STOP_CRIT)
        self.register_buffer('grammian', grammian)
        self.register_buffer('sol', sol)
        self.register_buffer('new_sol', new_sol)

    @torch.no_grad()
    def line_solver(self, v1v1, v1v2, v2v2):
        """Analytical solution for the min-norm problem
        """
        if v1v2 >= v1v1:
            return 0.999
        if v1v2 >= v2v2:
            return 0.001
        return (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)

    @torch.no_grad()
    def forward(self, vecs):
        """Computes grammian matrix G_{i,j} = (<v_i, v_j>)_{i,j}.
        """
        if self.n_tasks == 1:
            return vecs[0]
        if self.n_tasks == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            gamma = self.line_solver(v1v1, v1v2, v2v2)
            return gamma * vecs[0] + (1.0 - gamma) * vecs[1]
        self.sol.fill_(1.0 / self.n)
        self.new_sol.copy_(self.sol)
        torch.mm(vecs, vecs.t(), out=self.grammian)
        for iter_count in range(self.MAX_ITER):
            gram_dot_sol = torch.mv(self.grammian, self.sol)
            t_iter = torch.argmin(gram_dot_sol)
            v1v1 = torch.dot(self.sol, gram_dot_sol)
            v1v2 = torch.dot(self.sol, self.grammian[:, t_iter])
            v2v2 = self.grammian[t_iter, t_iter]
            gamma = self.line_solver(v1v1, v1v2, v2v2)
            self.new_sol *= gamma
            self.new_sol[t_iter] += 1.0 - gamma
            change = self.new_sol - self.sol
            if torch.sum(torch.abs(change)) < self.STOP_CRIT:
                return self.new_sol
            self.sol.copy_(self.new_sol)
        return self.sol


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNormPillow,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Block,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MinNormLinearSolver,
     lambda: ([], {}),
     lambda: ([0, 0, torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinNormPlanarSolver,
     lambda: ([], {'n_tasks': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MinNormSolver,
     lambda: ([], {'n_tasks': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MinNormSolverFW,
     lambda: ([], {'n_tasks': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hav4ik_Hydra(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

