import sys
_module = sys.modules[__name__]
del sys
api = _module
classy_train = _module
losses = _module
nbdt_losses = _module
main = _module
nbdt = _module
analysis = _module
data = _module
ade20k = _module
cifar = _module
custom = _module
imagenet = _module
lip = _module
pascal_context = _module
transforms = _module
graph = _module
hierarchy = _module
loss = _module
metrics = _module
model = _module
models = _module
resnet = _module
utils = _module
wideresnet = _module
nx = _module
wn = _module
tree = _module
utils = _module
setup = _module
tests = _module
conftest = _module
test_inference = _module
test_train = _module

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


from torchvision import set_image_backend


from torchvision import set_video_backend


import torch.nn as nn


import numpy as np


from torch import nn


from torch import optim


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


from torch.distributions import Categorical


from collections import defaultdict


import functools


import time


import random


from torch.nn import functional as F


from torch.utils import data


from torch.utils.data import Dataset


import torchvision.datasets as datasets


import math


import torch.utils.data as data


from sklearn.cluster import AgglomerativeClustering


from math import log


from torch.hub import load_state_dict_from_url


import torch.nn.init as init


def synset_to_name(synset):
    return synset.name().split('.')[0]


class FakeSynset:

    def __init__(self, wnid):
        self.wnid = wnid
        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset('f{:08d}'.format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return 'f'

    def name(self):
        return '(generated)'

    def definition(self):
        return '(generated)'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]
    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)


def wnid_to_name(wnid):
    return synset_to_name(wnid_to_synset(wnid))


class Node:

    def __init__(self, tree, wnid, other_class=False):
        self.tree = tree
        self.wnid = wnid
        self.name = wnid_to_name(wnid)
        self.synset = wnid_to_synset(wnid)
        self.original_classes = tree.classes
        self.num_original_classes = len(self.tree.wnids_leaves)
        self.has_other = other_class and not (self.is_root() or self.is_leaf())
        self.num_children = len(self.succ)
        self.num_classes = self.num_children + int(self.has_other)
        self.class_index_to_child_index, self.child_index_to_class_index = self.build_class_mappings()
        self.classes = self.build_classes()
        assert len(self.classes) == self.num_classes, f'Number of classes {self.num_classes} does not equal number of class names found ({len(self.classes)}): {self.classes}'
        self.leaves = list(self.get_leaves())
        self.num_leaves = len(self.leaves)

    def wnid_to_class_index(self, wnid):
        return self.tree.wnids_leaves.index(wnid)

    def wnid_to_child_index(self, wnid):
        return [child.wnid for child in self.children].index(wnid)

    @property
    def parent(self):
        if not self.parents:
            return None
        return self.parents[0]

    @property
    def pred(self):
        return self.tree.G.pred[self.wnid]

    @property
    def parents(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.pred]

    @property
    def succ(self):
        return self.tree.G.succ[self.wnid]

    @property
    def children(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.succ]

    def get_leaves(self):
        return get_leaves(self.tree.G, self.wnid)

    def is_leaf(self):
        return len(self.succ) == 0

    def is_root(self):
        return len(self.pred) == 0

    def build_class_mappings(self):
        if self.is_leaf():
            return {}, {}
        old_to_new = defaultdict(lambda : [])
        new_to_old = defaultdict(lambda : [])
        for new_index, child in enumerate(self.succ):
            for leaf in get_leaves(self.tree.G, child):
                old_index = self.wnid_to_class_index(leaf)
                old_to_new[old_index].append(new_index)
                new_to_old[new_index].append(old_index)
        if not self.has_other:
            return old_to_new, new_to_old
        new_index = self.num_children
        for old in range(self.num_original_classes):
            if old not in old_to_new:
                old_to_new[old].append(new_index)
                new_to_old[new_index].append(old)
        return old_to_new, new_to_old

    def build_classes(self):
        return [','.join([self.original_classes[old] for old in old_indices]) for new_index, old_indices in sorted(self.child_index_to_class_index.items(), key=lambda t: t[0])]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.child_index_to_class_index]

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


def fwd():
    """Get file's working directory"""
    return Path(__file__).parent.absolute()


def hierarchy_to_path_graph(dataset, hierarchy):
    return os.path.join(fwd(), f'hierarchies/{dataset}/graph-{hierarchy}.json')


def dataset_to_default_path_graph(dataset):
    return hierarchy_to_path_graph(dataset, 'induced')


def dataset_to_default_path_wnids(dataset):
    return os.path.join(fwd(), f'wnids/{dataset}.txt')


DATASETS = 'CIFAR10', 'CIFAR100', 'TinyImagenet200', 'Imagenet1000', 'Cityscapes', 'PascalContext', 'LookIntoPerson', 'ADE20K'


DATASET_TO_NUM_CLASSES = {'CIFAR10': 10, 'CIFAR100': 100, 'TinyImagenet200': 200, 'Imagenet1000': 1000, 'Cityscapes': 19, 'PascalContext': 59, 'LookIntoPerson': 20, 'ADE20K': 150}


def dataset_to_dummy_classes(dataset):
    assert dataset in DATASETS
    num_classes = DATASET_TO_NUM_CLASSES[dataset]
    return [FakeSynset.create_from_offset(i).wnid for i in range(num_classes)]


def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


def is_leaf(G, node):
    return len(G.succ[node]) == 0


def get_leaf_to_path(G):
    leaf_to_path = {}
    for root in get_roots(G):
        frontier = [(root, 0, [])]
        while frontier:
            node, child_index, path = frontier.pop(0)
            path = path + [(child_index, node)]
            if is_leaf(G, node):
                leaf_to_path[node] = path
                continue
            frontier.extend([(child, i, path) for i, child in enumerate(G.succ[node])])
    return leaf_to_path


def get_wnids(path_wnids):
    if not os.path.exists(path_wnids):
        parent = Path(fwd()).parent
        None
        path_wnids = parent / path_wnids
    with open(path_wnids) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]
    return wnids


def read_graph(path):
    if not os.path.exists(path):
        parent = Path(fwd()).parent
        None
        path = parent / path
    with open(path) as f:
        return node_link_graph(json.load(f))


class Tree:

    def __init__(self, dataset, path_graph=None, path_wnids=None, classes=None, hierarchy=None):
        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)
        self.load_hierarchy(dataset, path_graph, path_wnids, classes)

    def load_hierarchy(self, dataset, path_graph, path_wnids, classes):
        self.dataset = dataset
        self.path_graph = path_graph
        self.path_wnids = path_wnids
        self.classes = classes
        self.G = read_graph(path_graph)
        self.wnids_leaves = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids_leaves, self.classes)}
        self.wnid_to_class_index = {wnid: i for i, wnid in enumerate(self.wnids_leaves)}
        self.wnid_to_node = self.get_wnid_to_node()
        self.nodes = [self.wnid_to_node[wnid] for wnid in sorted(self.wnid_to_node)]
        self.inodes = [node for node in self.nodes if not node.is_leaf()]
        self.leaves = [self.wnid_to_node[wnid] for wnid in self.wnids_leaves]

    def update_from_model(self, model, arch, dataset, classes=None, path_wnids=None, path_graph=None):
        assert model is not None, '`model` cannot be NoneType'
        path_graph = generate_hierarchy(dataset=dataset, method='induced', arch=arch, model=model, path=path_graph)
        tree = Tree(dataset, path_graph=path_graph, path_wnids=path_wnids, classes=classes, hierarchy='induced')
        self.load_hierarchy(dataset=tree.dataset, path_graph=tree.path_graph, path_wnids=tree.path_wnids, classes=tree.classes)

    @classmethod
    def create_from_args(cls, args, classes=None):
        return cls(args.dataset, args.path_graph, args.path_wnids, classes=classes, hierarchy=args.hierarchy)

    @property
    def root(self):
        for node in self.inodes:
            if node.is_root():
                return node
        raise UserWarning('Should not be reachable. Tree should always have root')

    def get_wnid_to_node(self):
        wnid_to_node = {}
        for wnid in self.G:
            wnid_to_node[wnid] = Node(self, wnid)
        return wnid_to_node

    def get_leaf_to_steps(self):
        node = self.inodes[0]
        leaf_to_path = get_leaf_to_path(self.G)
        leaf_to_steps = {}
        for leaf in self.wnids_leaves:
            next_indices = [index for index, _ in leaf_to_path[leaf][1:]] + [-1]
            leaf_to_steps[leaf] = [{'node': self.wnid_to_node[wnid], 'name': self.wnid_to_node[wnid].name, 'next_index': next_index} for next_index, (_, wnid) in zip(next_indices, leaf_to_path[leaf])]
        return leaf_to_steps

    def visualize(self, path_html, dataset=None, **kwargs):
        """
        :param path_html: Where to write the final generated visualization
        """
        generate_hierarchy_vis_from(self.G, dataset=dataset, path_html=path_html, **kwargs)


class EmbeddedDecisionRules(nn.Module):

    def __init__(self, dataset=None, path_graph=None, path_wnids=None, classes=(), hierarchy=None, tree=None):
        super().__init__()
        if not tree:
            tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.tree = tree
        self.correct = 0
        self.total = 0
        self.I = torch.eye(len(self.tree.classes))

    @staticmethod
    def get_node_logits(outputs, node=None, new_to_old_classes=None, num_classes=None):
        """Get output for a particular node

        This `outputs` above are the output of the neural network.
        """
        assert node or new_to_old_classes and num_classes, 'Either pass node or (new_to_old_classes mapping and num_classes)'
        new_to_old_classes = new_to_old_classes or node.child_index_to_class_index
        num_classes = num_classes or node.num_classes
        return torch.stack([outputs.T[new_to_old_classes[child_index]].mean(dim=0) for child_index in range(num_classes)]).T

    @classmethod
    def get_all_node_outputs(cls, outputs, nodes):
        """Run hard embedded decision rules.

        Returns the output for *every single node.
        """
        wnid_to_outputs = {}
        for node in nodes:
            node_logits = cls.get_node_logits(outputs, node)
            node_outputs = {'logits': node_logits}
            if len(node_logits.size()) > 1:
                node_outputs['preds'] = torch.max(node_logits, dim=1)[1]
                node_outputs['probs'] = F.softmax(node_logits, dim=1)
                node_outputs['entropy'] = Categorical(probs=node_outputs['probs']).entropy()
            wnid_to_outputs[node.wnid] = node_outputs
        return wnid_to_outputs

    def forward_nodes(self, outputs):
        return self.get_all_node_outputs(outputs, self.tree.inodes)


class HardEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def get_node_logits_filtered(cls, node, outputs, targets):
        """'Smarter' inference for a hard node.

        If you have targets for the node, you can selectively perform inference,
        only for nodes where the label of a sample is well-defined.
        """
        classes = [node.class_index_to_child_index[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        targets_sub = [cls[0] for cls in classes if cls]
        outputs = outputs[selector]
        if outputs.size(0) == 0:
            return selector, outputs[:, :node.num_classes], targets_sub
        outputs_sub = cls.get_node_logits(outputs, node)
        return selector, outputs_sub, targets_sub

    @classmethod
    def traverse_tree(cls, wnid_to_outputs, tree):
        """Convert node outputs to final prediction.

        Note that the prediction output for this function can NOT be trained
        on. The outputs have been detached from the computation graph.
        """
        example = wnid_to_outputs[tree.inodes[0].wnid]
        n_samples = int(example['logits'].size(0))
        device = example['logits'].device
        for wnid in tuple(wnid_to_outputs.keys()):
            outputs = wnid_to_outputs[wnid]
            outputs['preds'] = list(map(int, outputs['preds'].cpu()))
            outputs['probs'] = outputs['probs'].detach().cpu()
        decisions = []
        preds = []
        for index in range(n_samples):
            decision = [{'node': tree.root, 'name': 'root', 'prob': 1, 'entropy': 0}]
            node = tree.root
            while not node.is_leaf():
                if node.wnid not in wnid_to_outputs:
                    node = None
                    break
                outputs = wnid_to_outputs[node.wnid]
                index_child = outputs['preds'][index]
                prob_child = float(outputs['probs'][index][index_child])
                node = node.children[index_child]
                decision.append({'node': node, 'name': node.name, 'prob': prob_child, 'next_index': index_child, 'entropy': float(outputs['entropy'][index])})
            preds.append(tree.wnid_to_class_index[node.wnid])
            decisions.append(decision)
        return torch.Tensor(preds).long(), decisions

    def predicted_to_logits(self, predicted):
        """Convert predicted classes to one-hot logits."""
        if self.I.device != predicted.device:
            self.I = self.I
        return self.I[predicted]

    def forward_with_decisions(self, outputs):
        wnid_to_outputs = self.forward_nodes(outputs)
        predicted, decisions = self.traverse_tree(wnid_to_outputs, self.tree)
        logits = self.predicted_to_logits(predicted)
        logits._nbdt_output_flag = True
        return logits, decisions

    def forward(self, outputs):
        outputs, _ = self.forward_with_decisions(outputs)
        return outputs


class TreeSupLoss(nn.Module):
    accepts_tree = lambda tree, **kwargs: tree
    accepts_criterion = lambda criterion, **kwargs: criterion
    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_tree_supervision_weight = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes
    accepts_hierarchy = True
    accepts_tree_supervision_weight_end = True
    accepts_tree_supervision_weight_power = True
    accepts_xent_weight = True
    accepts_xent_weight_end = True
    accepts_xent_weight_power = True

    def __init__(self, dataset, criterion, path_graph=None, path_wnids=None, classes=None, hierarchy=None, Rules=HardEmbeddedDecisionRules, tree=None, tree_supervision_weight=1.0, tree_supervision_weight_end=None, tree_supervision_weight_power=1, xent_weight=1, xent_weight_end=None, xent_weight_power=1):
        super().__init__()
        if not tree:
            tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.num_classes = len(tree.classes)
        self.tree = tree
        self.rules = Rules(tree=tree)
        self.tree_supervision_weight = tree_supervision_weight
        self.tree_supervision_weight_end = tree_supervision_weight_end if tree_supervision_weight_end is not None else tree_supervision_weight
        self.tree_supervision_weight_power = tree_supervision_weight_power
        self.xent_weight = xent_weight
        self.xent_weight_end = xent_weight_end if xent_weight_end is not None else xent_weight
        self.xent_weight_power = xent_weight_power
        self.criterion = criterion
        self.progress = 1
        self.epochs = 0

    @staticmethod
    def assert_output_not_nbdt(outputs):
        """
        >>> x = torch.randn(1, 3, 224, 224)
        >>> TreeSupLoss.assert_output_not_nbdt(x)  # all good!
        >>> x._nbdt_output_flag = True
        >>> TreeSupLoss.assert_output_not_nbdt(x)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        >>> from nbdt.model import NBDT
        >>> import torchvision.models as models
        >>> model = models.resnet18()
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  # all good!
        >>> model = NBDT('CIFAR10', model, arch='ResNet18')
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        """
        assert getattr(outputs, '_nbdt_output_flag', False) is False, "Uh oh! Looks like you passed an NBDT model's output to an NBDT loss. NBDT losses are designed to take in the *original* model's outputs, as input. NBDT models are designed to only be used during validation and inference, not during training. Confused?  Check out github.com/alvinwan/nbdt#convert-neural-networks-to-decision-trees for examples and instructions."

    def forward_tree(self, outputs, targets):
        raise NotImplementedError()

    def get_weight(self, start, end, power=1):
        progress = self.progress ** power
        return (1 - progress) * start + progress * end

    def forward(self, outputs, targets):
        loss_xent = self.criterion(outputs, targets)
        loss_tree = self.forward_tree(outputs, targets)
        tree_weight = self.get_weight(self.tree_supervision_weight, self.tree_supervision_weight_end, self.tree_supervision_weight_power)
        xent_weight = self.get_weight(self.xent_weight, self.xent_weight_end, self.xent_weight_power)
        return loss_xent * xent_weight + loss_tree * tree_weight

    def set_epoch(self, cur, total):
        self.epochs = cur
        self.progress = cur / total
        if hasattr(super(), 'set_epoch'):
            super().set_epoch(cur, total)


class HardTreeSupLoss(TreeSupLoss):

    def forward_tree(self, outputs, targets):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.

        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        self.assert_output_not_nbdt(outputs)
        loss = 0
        num_losses = outputs.size(0) * len(self.tree.inodes) / 2.0
        outputs_subs = defaultdict(lambda : [])
        targets_subs = defaultdict(lambda : [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.tree.inodes:
            _, outputs_sub, targets_sub = HardEmbeddedDecisionRules.get_node_logits_filtered(node, outputs, targets_ints)
            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)
        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long()
            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) * self.tree_supervision_weight
            loss += self.criterion(outputs_sub, targets_sub) * fraction
        return loss


class SoftEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def traverse_tree(cls, wnid_to_outputs, tree):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.child_index_to_class_index[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run. The output is a single distribution over all leaves. The
        ordering is determined by the original ordering of the provided logits.
        (I think. Need to check nbdt.data.custom.Node)
        """
        example = wnid_to_outputs[tree.inodes[0].wnid]
        num_samples = example['logits'].size(0)
        num_classes = len(tree.classes)
        device = example['logits'].device
        class_probs = torch.ones((num_samples, num_classes))
        for node in tree.inodes:
            outputs = wnid_to_outputs[node.wnid]
            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.child_index_to_class_index[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))
            assert len(set(old_indices)) == len(old_indices), 'All old indices must be unique in order for this operation to be correct.'
            class_probs[:, old_indices] *= outputs['probs'][:, new_indices]
        return class_probs

    def forward_with_decisions(self, outputs):
        wnid_to_outputs = self.forward_nodes(outputs)
        outputs = self.forward(outputs, wnid_to_outputs)
        _, predicted = outputs.max(1)
        decisions = []
        node = self.tree.inodes[0]
        leaf_to_steps = self.tree.get_leaf_to_steps()
        for index, prediction in enumerate(predicted):
            leaf = self.tree.wnids_leaves[prediction]
            steps = leaf_to_steps[leaf]
            probs = [1]
            entropies = [0]
            for step in steps[:-1]:
                _out = wnid_to_outputs[step['node'].wnid]
                _probs = _out['probs'][0]
                probs.append(_probs[step['next_index']])
                entropies.append(Categorical(probs=_probs).entropy().item())
            for step, prob, entropy in zip(steps, probs, entropies):
                step['prob'] = float(prob)
                step['entropy'] = float(entropy)
            decisions.append(steps)
        return outputs, decisions

    def forward(self, outputs, wnid_to_outputs=None):
        if not wnid_to_outputs:
            wnid_to_outputs = self.forward_nodes(outputs)
        logits = self.traverse_tree(wnid_to_outputs, self.tree)
        logits._nbdt_output_flag = True
        return logits


class SoftTreeSupLoss(TreeSupLoss):

    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftEmbeddedDecisionRules, **kwargs)

    def forward_tree(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)
        return self.criterion(self.rules(outputs), targets)


class SoftTreeLoss(SoftTreeSupLoss):
    accepts_tree_start_epochs = True
    accepts_tree_update_every_epochs = True
    accepts_tree_update_end_epochs = True
    accepts_arch = True
    accepts_net = lambda net, **kwargs: net
    accepts_checkpoint_path = lambda checkpoint_path, **kwargs: checkpoint_path

    def __init__(self, *args, arch=None, checkpoint_path='./', net=None, tree_start_epochs=67, tree_update_every_epochs=10, tree_update_end_epochs=120, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epochs = tree_start_epochs
        self.update_every_epochs = tree_update_every_epochs
        self.update_end_epochs = tree_update_end_epochs
        self.net = net
        self.arch = arch
        self.checkpoint_path = checkpoint_path

    def forward_tree(self, outputs, targets):
        if self.epochs < self.start_epochs:
            return self.criterion(outputs, targets)
        self.assert_output_not_nbdt(outputs)
        return self.criterion(self.rules(outputs), targets)

    def set_epoch(self, *args, **kwargs):
        super().set_epoch(*args, **kwargs)
        offset = self.epochs - self.start_epochs
        if offset >= 0 and offset % self.update_every_epochs == 0 and self.epochs < self.update_end_epochs:
            checkpoint_dir = self.checkpoint_path.replace('.pth', '')
            path_graph = os.path.join(checkpoint_dir, f'graph-epoch{self.epochs}.json')
            self.tree.update_from_model(self.net, self.arch, self.tree.dataset, path_graph=path_graph)


def coerce_tensor(x, is_label=False):
    if is_label:
        return x.reshape(-1, 1)
    else:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def uncoerce_tensor(x, original_shape):
    n, c, h, w = original_shape
    return x.reshape(n, h, w, c).permute(0, 3, 1, 2)


class SoftSegTreeSupLoss(SoftTreeSupLoss):

    def forward(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)
        loss = self.criterion(outputs, targets)
        coerced_outputs = coerce_tensor(outputs)
        bayesian_outputs = self.rules(coerced_outputs)
        bayesian_outputs = uncoerce_tensor(bayesian_outputs, outputs.shape)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss


def coerce_state_dict(state_dict, reference_state_dict):
    if 'net' in state_dict:
        state_dict = state_dict['net']
    has_reference_module = list(reference_state_dict)[0].startswith('module.')
    has_module = list(state_dict)[0].startswith('module.')
    if not has_reference_module and has_module:
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    elif has_reference_module and not has_module:
        state_dict = {('module.' + key): value for key, value in state_dict.items()}
    return state_dict


class Colors:
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    ENDC = '\x1b[0m'
    BOLD = '\x1b[1m'
    CYAN = '\x1b[36m'

    @classmethod
    def red(cls, *args):
        None

    @classmethod
    def green(cls, *args):
        None

    @classmethod
    def cyan(cls, *args):
        None

    @classmethod
    def bold(cls, *args):
        None


def load_state_dict_from_key(keys, model_urls, pretrained=False, progress=True, root='.cache/torch/checkpoints', device='cpu'):
    valid_keys = [key for key in keys if key in model_urls]
    if not valid_keys:
        raise UserWarning(f'None of the keys {keys} correspond to a pretrained model.')
    key = valid_keys[-1]
    url = model_urls[key]
    Colors.green(f'Loading pretrained model {key} from {url}')
    return load_state_dict_from_url(url, Path.home() / root, progress=progress, check_hash=False, map_location=torch.device(device))


model_urls = {('wrn28_10', 'TinyImagenet200'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-wrn28_10.pth'}


class NBDT(nn.Module):

    def __init__(self, dataset, model, arch=None, path_graph=None, path_wnids=None, classes=None, hierarchy=None, pretrained=None, **kwargs):
        super().__init__()
        if dataset and not hierarchy and not path_graph:
            assert arch, 'Must specify `arch` if no `hierarchy` or `path_graph`'
            hierarchy = f'induced-{arch}'
        if pretrained and not arch:
            raise UserWarning('To load a pretrained NBDT, you need to specify the `arch`. `arch` is the name of the architecture. e.g., ResNet18')
        if isinstance(model, str):
            raise NotImplementedError('Model must be nn.Module')
        tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.init(dataset, model, tree, arch=arch, pretrained=pretrained, hierarchy=hierarchy, **kwargs)

    def init(self, dataset, model, tree, arch=None, pretrained=False, hierarchy=None, eval=True, Rules=HardEmbeddedDecisionRules):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.rules = Rules(tree=tree)
        self.model = model
        if pretrained:
            assert arch is not None
            keys = [(arch, dataset), (arch, dataset, hierarchy)]
            state_dict = load_state_dict_from_key(keys, model_urls, pretrained=True)
            self.load_state_dict(state_dict)
        if eval:
            self.eval()

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = coerce_state_dict(state_dict, self.model.state_dict())
        return self.model.load_state_dict(state_dict, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def forward(self, x):
        x = self.model(x)
        x = self.rules(x)
        return x

    def forward_with_decisions(self, x):
        x = self.model(x)
        x, decisions = self.rules.forward_with_decisions(x)
        return x, decisions


class HardNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({'Rules': HardEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SoftNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({'Rules': SoftEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SegNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        assert len(x.shape) == 4, 'Input must be of shape (N,C,H,W) for segmentation'
        x = self.model(x)
        original_shape = x.shape
        x = coerce_tensor(x)
        x = self.rules.forward(x)
        x = uncoerce_tensor(x, original_shape)
        return x


class HardSegNBDT(SegNBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({'Rules': HardEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SoftSegNBDT(SegNBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({'Rules': SoftEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_alvinwan_neural_backed_decision_trees(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

