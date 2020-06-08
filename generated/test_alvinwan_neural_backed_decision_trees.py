import sys
_module = sys.modules[__name__]
del sys
api = _module
main = _module
nbdt = _module
analysis = _module
data = _module
custom = _module
imagenet = _module
graph = _module
hierarchy = _module
loss = _module
model = _module
models = _module
resnet = _module
utils = _module
wideresnet = _module
utils = _module
setup = _module
conftest = _module
test_inference = _module
test_train = _module

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


import torch.optim as optim


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import numpy as np


from torch.utils.data import Dataset


from collections import defaultdict


import random


import math


import torch.nn.init as init


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


DATASET_TO_NUM_CLASSES = {'CIFAR10': 10, 'CIFAR100': 100, 'TinyImagenet200':
    200, 'Imagenet1000': 1000}


DATASETS = 'CIFAR10', 'CIFAR100', 'TinyImagenet200', 'Imagenet1000'


def dataset_to_dummy_classes(dataset):
    assert dataset in DATASETS
    num_classes = DATASET_TO_NUM_CLASSES[dataset]
    return [FakeSynset.create_from_offset(i).wnid for i in range(num_classes)]


def fwd():
    """Get file's working directory"""
    return Path(__file__).parent.absolute()


def dataset_to_default_path_wnids(dataset):
    return os.path.join(fwd(), f'wnids/{dataset}.txt')


def synset_to_name(synset):
    return synset.name().split('.')[0]


def wnid_to_synset(wnid):
    from nltk.corpus import wordnet as wn
    offset = int(wnid[1:])
    pos = wnid[0]
    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)


def wnid_to_name(wnid):
    return synset_to_name(wnid_to_synset(wnid))


def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


def get_root(G):
    roots = list(get_roots(G))
    assert len(roots) == 1, f'Multiple ({len(roots)}) found'
    return roots[0]


def hierarchy_to_path_graph(dataset, hierarchy):
    return os.path.join(fwd(), f'hierarchies/{dataset}/graph-{hierarchy}.json')


def dataset_to_default_path_graph(dataset):
    return hierarchy_to_path_graph(dataset, 'induced')


def get_non_leaves(G):
    for node in G.nodes:
        if len(G.succ[node]) > 0:
            yield node


def read_graph(path):
    if not os.path.exists(path):
        parent = Path(fwd()).parent
        print(f'No such file or directory: {path}. Looking in {str(parent)}')
        path = parent / path
    with open(path) as f:
        return node_link_graph(json.load(f))


def get_wnids(path_wnids):
    if not os.path.exists(path_wnids):
        parent = Path(fwd()).parent
        print(
            f'No such file or directory: {path_wnids}. Looking in {str(parent)}'
            )
        path_wnids = parent / path_wnids
    with open(path_wnids) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]
    return wnids


class Node:

    def __init__(self, wnid, classes, path_graph, path_wnids, other_class=False
        ):
        self.path_graph = path_graph
        self.path_wnids = path_wnids
        self.wnid = wnid
        self.wnids = get_wnids(path_wnids)
        self.G = read_graph(path_graph)
        self.synset = wnid_to_synset(wnid)
        self.original_classes = classes
        self.num_original_classes = len(self.wnids)
        assert not self.is_leaf(), 'Cannot build dataset for leaf'
        self.has_other = other_class and not (self.is_root() or self.is_leaf())
        self.num_children = len(self.get_children())
        self.num_classes = self.num_children + int(self.has_other)
        self.old_to_new_classes, self.new_to_old_classes = (self.
            build_class_mappings())
        self.classes = self.build_classes()
        assert len(self.classes
            ) == self.num_classes, f'Number of classes {self.num_classes} does not equal number of class names found ({len(self.classes)}): {self.classes}'
        self.children = list(self.get_children())
        self.leaves = list(self.get_leaves())
        self.num_leaves = len(self.leaves)
        self._probabilities = None
        self._class_weights = None

    def wnid_to_class_index(self, wnid):
        return self.wnids.index(wnid)

    def get_parents(self):
        return self.G.pred[self.wnid]

    def get_children(self):
        return self.G.succ[self.wnid]

    def get_leaves(self):
        return get_leaves(self.G, self.wnid)

    def is_leaf(self):
        return len(self.get_children()) == 0

    def is_root(self):
        return len(self.get_parents()) == 0

    def build_class_mappings(self):
        old_to_new = defaultdict(lambda : [])
        new_to_old = defaultdict(lambda : [])
        for new_index, child in enumerate(self.get_children()):
            for leaf in get_leaves(self.G, child):
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
        return [','.join([self.original_classes[old] for old in old_indices
            ]) for new_index, old_indices in sorted(self.new_to_old_classes
            .items(), key=lambda t: t[0])]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.new_to_old_classes]

    @property
    def probabilities(self):
        """Calculates probability of training on the ith class.

        If the class contains more than `resample_threshold` samples, the
        probability is lower, as it is likely to cause severe class imbalance
        issues.
        """
        if self._probabilities is None:
            reference = min(self.class_counts)
            self._probabilities = torch.Tensor([min(1, reference / len(
                old_indices)) for old_indices in self.new_to_old_classes])
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def class_weights(self):
        if self._class_weights is None:
            self._class_weights = self.probabilities
        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights):
        self._class_weights = class_weights

    @staticmethod
    def get_wnid_to_node(path_graph, path_wnids, classes):
        wnid_to_node = {}
        G = read_graph(path_graph)
        for wnid in get_non_leaves(G):
            wnid_to_node[wnid] = Node(wnid, classes, path_graph=path_graph,
                path_wnids=path_wnids)
        return wnid_to_node

    @staticmethod
    def get_nodes(path_graph, path_wnids, classes):
        wnid_to_node = Node.get_wnid_to_node(path_graph, path_wnids, classes)
        wnids = sorted(wnid_to_node)
        nodes = [wnid_to_node[wnid] for wnid in wnids]
        return nodes

    @staticmethod
    def get_leaf_to_path(nodes):
        node = nodes[0]
        leaf_to_path = get_leaf_to_path(node.G)
        wnid_to_node = {node.wnid: node for node in nodes}
        leaf_to_path_nodes = {}
        for leaf in leaf_to_path:
            leaf_to_path_nodes[leaf] = [{'node': wnid_to_node.get(wnid,
                None), 'name': wnid_to_name(wnid)} for wnid in leaf_to_path
                [leaf]]
        return leaf_to_path_nodes

    @staticmethod
    def get_root_node_wnid(path_graph):
        raise UserWarning('Root node may have wnid now')
        tree = ET.parse(path_graph)
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is not None:
                return wnid
        return None

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class EmbeddedDecisionRules(nn.Module):

    def __init__(self, dataset, path_graph=None, path_wnids=None, classes=()):
        if not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if not classes:
            classes = dataset_to_dummy_classes(dataset)
        super().__init__()
        assert all([dataset, path_graph, path_wnids, classes])
        self.classes = classes
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}
        self.wnids = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids,
            self.classes)}
        self.correct = 0
        self.total = 0
        self.I = torch.eye(len(classes))

    @staticmethod
    def get_node_logits(outputs, node):
        """Get output for a particular node

        This `outputs` above are the output of the neural network.
        """
        return torch.stack([outputs.T[node.new_to_old_classes[new_label]].
            mean(dim=0) for new_label in range(node.num_classes)]).T

    @classmethod
    def get_all_node_outputs(cls, outputs, nodes):
        """Run hard embedded decision rules.

        Returns the output for *every single node.
        """
        wnid_to_outputs = {}
        for node in nodes:
            node_logits = cls.get_node_logits(outputs, node)
            wnid_to_outputs[node.wnid] = {'logits': node_logits, 'preds':
                torch.max(node_logits, dim=1)[1], 'probs': F.softmax(
                node_logits, dim=1)}
        return wnid_to_outputs

    def forward_nodes(self, outputs):
        return self.get_all_node_outputs(outputs, self.nodes)


model_urls = {('wrn28_10', 'TinyImagenet200'):
    'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-wrn28_10.pth'
    }


def load_state_dict_from_key(keys, model_urls, pretrained=False, progress=
    True, root='.cache/torch/checkpoints', device='cpu'):
    valid_keys = [key for key in keys if key in model_urls]
    if not valid_keys:
        raise UserWarning(
            f'None of the keys {keys} correspond to a pretrained model.')
    return load_state_dict_from_url(model_urls[valid_keys[-1]], Path.home() /
        root, progress=progress, check_hash=False, map_location=torch.
        device(device))


def coerce_state_dict(state_dict, reference_state_dict):
    if 'net' in state_dict:
        state_dict = state_dict['net']
    has_reference_module = list(reference_state_dict)[0].startswith('module.')
    has_module = list(state_dict)[0].startswith('module.')
    if not has_reference_module and has_module:
        state_dict = {key.replace('module.', '', 1): value for key, value in
            state_dict.items()}
    elif has_reference_module and not has_module:
        state_dict = {('module.' + key): value for key, value in state_dict
            .items()}
    return state_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
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

    def featurize(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.featurize(x)
        out = self.linear(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_alvinwan_neural_backed_decision_trees(_paritybench_base):
    pass

    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
