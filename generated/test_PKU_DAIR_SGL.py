import sys
_module = sys.modules[__name__]
del sys
gamlp_products = _module
nafs_link_predict = _module
nafs_node_cluster = _module
sgc_pubmed = _module
test_nas = _module
test_nas_dist = _module
test_nodeclass_dist = _module
setup = _module
sgl = _module
data = _module
base_data = _module
base_dataset = _module
utils = _module
dataset = _module
acm = _module
actor = _module
airports = _module
amazon = _module
amazon_product = _module
choose_edge_type = _module
coauthor = _module
dblp = _module
dblp_original = _module
facebook = _module
flickr = _module
github = _module
karateclub = _module
linkx_dataset = _module
nell = _module
ogbn = _module
ogbn_mag = _module
planetoid = _module
reddit = _module
twitch = _module
utils = _module
webkb = _module
wikics = _module
etc = _module
auto_select_edge_type_for_nars = _module
hetero_search = _module
hetero_test = _module
stability_of_subgraph_weight = _module
models = _module
base_model = _module
base_model_dist = _module
hetero = _module
fast_nars_sgc = _module
nars_sign = _module
homo = _module
gamlp = _module
gamlp_dist = _module
gamlp_recursive = _module
gbp = _module
nafs = _module
pasca_v1 = _module
pasca_v2 = _module
pasca_v3 = _module
sgc = _module
sgc_dist = _module
sign = _module
ssgc = _module
simple_models = _module
operators = _module
base_op = _module
graph_op = _module
laplacian_graph_op = _module
ppr_graph_op = _module
message_op = _module
concat_message_op = _module
iterate_learnable_weighted_message_op = _module
last_message_op = _module
learnable_weighted_messahe_op = _module
max_message_op = _module
mean_message_op = _module
min_message_op = _module
over_smooth_distance_op = _module
projected_concat_message_op = _module
simple_weighted_message_op = _module
sum_message_op = _module
utils = _module
search = _module
auto_search = _module
auto_search_dist = _module
base_search = _module
search_config_dist = _module
search_models = _module
search_models_dist = _module
utils = _module
tasks = _module
base_task = _module
clustering_metrics = _module
correct_and_smooth = _module
link_prediction = _module
node_classification = _module
node_classification_dist = _module
node_classification_with_label_use = _module
node_clustering = _module
utils = _module
tricks = _module
correct_and_smooth = _module
utils = _module
auto_choose_gpu = _module

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


import numpy as np


from scipy.sparse import csr_matrix


from torch import Tensor


import itertools


import warnings


from typing import Tuple


import scipy.sparse as sp


from itertools import product


from typing import Callable


from typing import List


from typing import Optional


import torch.nn.functional as F


from scipy.io import loadmat


from itertools import chain


import time


from functools import reduce


from typing import Dict


import torch.nn as nn


from torch.nn import Linear


from torch import nn


from torch.nn import Parameter


from torch.nn import ModuleList


import numpy.ctypeslib as ctl


from torch.optim import Adam


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import random


import torch.utils.data as Data


from sklearn.cluster import KMeans


import math


from sklearn.metrics import roc_auc_score


from sklearn.metrics import average_precision_score


class BaseSGAPModel(nn.Module):

    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim
        self._pre_graph_op, self._pre_msg_op = None, None
        self._post_graph_op, self._post_msg_op = None, None
        self._base_model = None
        self._processed_feat_list = None
        self._processed_feature = None
        self._pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self._pre_graph_op is not None:
            self._processed_feat_list = self._pre_graph_op.propagate(adj, feature)
            if self._pre_msg_op.aggr_type in ['proj_concat', 'learnable_weighted', 'iterate_learnable_weighted']:
                self._pre_msg_learnable = True
            else:
                self._pre_msg_learnable = False
                self._processed_feature = self._pre_msg_op.aggregate(self._processed_feat_list)
        else:
            self._pre_msg_learnable = False
            self._processed_feature = feature

    def postprocess(self, adj, output):
        if self._post_graph_op is not None:
            if self._post_msg_op.aggr_type in ['proj_concat', 'learnable_weighted', 'iterate_learnable_weighted']:
                raise ValueError('Learnable weighted message operator is not supported in the post-processing phase!')
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self._post_graph_op.propagate(adj, output)
            output = self._post_msg_op.aggregate(output)
        return output

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self._pre_msg_learnable is False:
            processed_feature = self._processed_feature[idx]
        else:
            transferred_feat_list = [feat[idx] for feat in self._processed_feat_list]
            processed_feature = self._pre_msg_op.aggregate(transferred_feat_list)
        output = self._base_model(processed_feature)
        return output


EDGE_TYPE_DELIMITER = '__to__'


def EdgeTypeStr2Tuple(edge_type: str) ->Tuple[str]:
    edge_type_list = edge_type.split(EDGE_TYPE_DELIMITER)
    return tuple(edge_type_list)


def ChooseEdgeType(edge_type_num: int, edge_types: List, predict_class: str) ->Tuple[str]:
    explored_node_type_set = {predict_class}
    chosen_edge_types_list = []
    candidate_edge_types_list = []
    other_edge_types_set = set(edge_types)
    for _ in range(edge_type_num):
        edge_types_to_move = [et for et in other_edge_types_set if len(set(EdgeTypeStr2Tuple(et)) & explored_node_type_set) > 0]
        candidate_edge_types_list += edge_types_to_move
        other_edge_types_set -= set(edge_types_to_move)
        if len(candidate_edge_types_list) == 0:
            warnings.warn(f"Can't find enough ({edge_type_num}) edge types!", UserWarning)
            break
        new_edge_type = random.choice(candidate_edge_types_list)
        chosen_edge_types_list.append(new_edge_type)
        candidate_edge_types_list.remove(new_edge_type)
        explored_node_type_set |= set(EdgeTypeStr2Tuple(new_edge_type))
    return tuple(sorted(chosen_edge_types_list))


def Combination(n: int, k: int) ->int:
    if n < 0 or k < 0:
        raise ValueError('n < 0 or k < 0!')
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def RemoveDuplicateEdgeType(edge_types: List) ->List[str]:
    unique_edge_types = []
    for et in edge_types:
        et_tuple = EdgeTypeStr2Tuple(et)
        reversed_et = et_tuple[1] + EDGE_TYPE_DELIMITER + et_tuple[0]
        if reversed_et not in unique_edge_types:
            unique_edge_types.append(et)
    return unique_edge_types


def ChooseMultiSubgraphs(subgraph_num: int, edge_type_num: int, edge_types: List, predict_class: str) ->List[Tuple[str]]:
    subgraph_edge_types_list = []
    unique_edge_type = RemoveDuplicateEdgeType(edge_types)
    if edge_type_num > len(unique_edge_type):
        return subgraph_edge_types_list
    maximal_reasonable_steps = 10 * Combination(len(unique_edge_type), edge_type_num) * (math.log2(Combination(len(unique_edge_type), edge_type_num)) + 1)
    step_cnt = 0
    for _ in range(subgraph_num):
        while True:
            step_cnt += 1
            if step_cnt > maximal_reasonable_steps:
                warnings.warn(f"Can't find enough ({subgraph_num}) subgraphs!", UserWarning)
                break
            new_subgraph_edge_types = ChooseEdgeType(edge_type_num, unique_edge_type, predict_class)
            if new_subgraph_edge_types in subgraph_edge_types_list:
                continue
            if len(new_subgraph_edge_types) > 0:
                subgraph_edge_types_list.append(new_subgraph_edge_types)
            break
    return subgraph_edge_types_list


class Edge:

    def __init__(self, row, col, edge_weight, edge_type, num_node, edge_attrs=None):
        if not isinstance(edge_type, str):
            raise TypeError('Edge type must be a string!')
        self.__edge_type = edge_type
        if not isinstance(row, (list, np.ndarray, Tensor)) or not isinstance(col, (list, np.ndarray, Tensor)) or not isinstance(edge_weight, (list, np.ndarray, Tensor)):
            raise TypeError('Row, col and edge_weight must be a list, np.ndarray or Tensor!')
        self.__row = row
        self.__col = col
        self.__edge_weight = edge_weight
        self.__edge_attrs = edge_attrs
        self.__num_edge = len(row)
        if isinstance(row, Tensor) or isinstance(col, Tensor):
            self.__sparse_matrix = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())), shape=(num_node, num_node))
        else:
            self.__sparse_matrix = csr_matrix((edge_weight, (row, col)), shape=(num_node, num_node))

    @property
    def sparse_matrix(self):
        return self.__sparse_matrix

    @property
    def edge_type(self):
        return self.__edge_type

    @property
    def num_edge(self):
        return self.__num_edge

    @property
    def edge_index(self):
        return self.__row, self.__col

    @property
    def edge_attrs(self):
        return self.__edge_attrs

    @edge_attrs.setter
    def edge_attrs(self, edge_attrs):
        self.__edge_attrs = edge_attrs

    @property
    def row(self):
        return self.__row

    @property
    def col(self):
        return self.__col

    @property
    def edge_weight(self):
        return self.__edge_weight


class Node:

    def __init__(self, node_type, num_node, x=None, y=None, node_ids=None):
        if not isinstance(num_node, int):
            raise TypeError('Num nodes must be a integer!')
        elif not isinstance(node_type, str):
            raise TypeError('Node type must be a string!')
        elif node_ids is not None and not isinstance(node_ids, (list, np.ndarray, Tensor)):
            raise TypeError('Node IDs must be a list, np.ndarray or Tensor!')
        self.__num_node = num_node
        self.__node_type = node_type
        if node_ids is not None:
            self.__node_ids = node_ids
        else:
            self.__node_ids = range(num_node)
        self.__x = x
        self.__y = y

    @property
    def num_node(self):
        return self.__num_node

    @property
    def node_ids(self):
        return self.__node_ids

    @property
    def node_type(self):
        return self.__node_type

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y


def file_exist(filepaths):
    if isinstance(filepaths, list):
        for filepath in filepaths:
            if not osp.exists(filepath):
                return False
        return True
    elif osp.exists(filepaths):
        return True
    else:
        return False


def to_undirected(edge_index):
    row, col = edge_index
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index


class HeteroNodeDataset:

    def __init__(self, root, name):
        self._name = name
        self._root = osp.join(root, name)
        self._raw_dir = osp.join(self._root, 'raw')
        self._processed_dir = osp.join(self._root, 'processed')
        self._data = None
        self._train_idx, self._val_idx, self._test_idx = None, None, None
        self.__preprocess()

    @property
    def name(self):
        return self._name

    @property
    def raw_file_paths(self):
        raise NotImplementedError

    @property
    def processed_file_paths(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _process(self):
        raise NotImplementedError

    def __preprocess(self):
        if file_exist(self.raw_file_paths):
            None
        else:
            None
            if not file_exist(self._raw_dir):
                os.makedirs(self._raw_dir)
            self._download()
            None
        if file_exist(self.processed_file_paths):
            None
        else:
            None
            if not file_exist(self._processed_dir):
                os.makedirs(self._processed_dir)
            self._process()
            None

    def __getitem__(self, key):
        if key in self.data.edge_types:
            return self.data[key]
        elif key in self.data.node_types:
            return self.data[key]
        else:
            raise ValueError('Please input valid edge type or node type!')

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Edge type or node type must be a string!')
        if key in self.data.edge_types:
            if not isinstance(value, Edge):
                raise TypeError('Please organize the dataset using the Edge class!')
            self.data.edges_dict[key] = value
        elif key in self.data.node_types:
            if not isinstance(value, Node):
                raise TypeError('Please organize the dataset using the Node class!')
            self.data.nodes_dict[key] = value
        else:
            raise ValueError('Please input valid edge type or node type!')

    @property
    def data(self):
        return self._data

    @property
    def y_dict(self):
        return self._data.y_dict

    @property
    def node_types(self):
        return self._data.node_types

    @property
    def edge_types(self):
        return self._data.edge_types

    @property
    def edge_type_cnt(self):
        return len(self.edge_types)

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    def sample_by_edge_type(self, edge_types, undirected=True):
        if not isinstance(edge_types, (str, list, tuple)):
            raise TypeError('The given edge types must be a string or a list or a tuple!')
        elif isinstance(edge_types, str):
            edge_types = [edge_types]
        elif isinstance(edge_types, (list, tuple)):
            for edge_type in edge_types:
                if not isinstance(edge_type, str):
                    raise TypeError('Edge type must be a string!')
        pre_sampled_node_types = []
        for edge_type in edge_types:
            pre_sampled_node_types = pre_sampled_node_types + [edge_type.split('__')[0], edge_type.split('__')[2]]
        pre_sampled_node_types = list(set(pre_sampled_node_types))
        sampled_node_types = []
        node_id_offsets = {}
        node_count = 0
        for node_type in self.node_types:
            if node_type in pre_sampled_node_types:
                sampled_node_types.append(node_type)
            node_id_offsets[node_type] = node_count
            node_count = node_count + self._data.num_node[node_type]
        num_node = 0
        feature = None
        node_id = None
        node_id_offset = {}
        for node_type in sampled_node_types:
            node_id_offset[node_type] = node_id_offsets[node_type] - num_node
            num_node = num_node + self._data.num_node[node_type]
            current_feature = torch.from_numpy(self._data[node_type].x)
            if current_feature is None:
                warnings.warn(f'{node_type} nodes have no features!', UserWarning)
            if feature is None:
                feature = current_feature
            else:
                feature = torch.vstack((feature, current_feature))
            if node_id is None:
                node_id = self._data.node_id_dict[node_type][:]
            else:
                node_id = node_id + self._data.node_id_dict[node_type]
        rows, cols = None, None
        for edge_type in edge_types:
            row_temp, col_temp = self._data[edge_type].edge_index
            node_type_of_row = edge_type.split('__')[0]
            node_type_of_col = edge_type.split('__')[2]
            row_temp = row_temp - node_id_offset[node_type_of_row]
            col_temp = col_temp - node_id_offset[node_type_of_col]
            if undirected is True and node_type_of_row != node_type_of_col:
                row_temp, col_temp = to_undirected((row_temp, col_temp))
            if rows is None:
                rows, cols = row_temp, col_temp
            else:
                rows = torch.hstack((rows, row_temp))
                cols = torch.hstack((cols, col_temp))
        edge_weight = torch.ones(len(rows))
        adj = csr_matrix((edge_weight.numpy(), (rows.numpy(), cols.numpy())), shape=(num_node, num_node))
        adj.data = torch.ones(len(adj.data)).numpy()
        return adj, feature.numpy(), torch.LongTensor(node_id)

    def sample_by_meta_path(self, meta_path, undirected=True):
        if isinstance(meta_path, str):
            if len(meta_path.split('__')) == 3:
                return self.sample_by_edge_type(meta_path, undirected)
        node_types = meta_path.split('__')
        node_type_st, node_type_ed = node_types[0], node_types[-1]
        sampled_node_types = []
        num_node = 0
        for node_type in self.node_types:
            if node_type in [node_type_st, node_type_ed]:
                sampled_node_types.append(node_type)
                num_node = num_node + self._data.num_node[node_type]
        feature = None
        node_id = None
        for node_type in sampled_node_types:
            current_feature = torch.from_numpy(self._data[node_type].x)
            if current_feature is None:
                warnings.warn(f'{node_type} nodes have no features!', UserWarning)
            if feature is None:
                feature = current_feature
            else:
                feature = torch.vstack((feature, current_feature))
            if node_id is None:
                node_id = self._data.node_id_dict[node_type][:]
            else:
                node_id = node_id + self._data.node_id_dict[node_type]
        adj = None
        for i in range(int((len(node_types) - 1) / 2)):
            edge_type = '__'.join([node_types[i * 2], 'to', node_types[(i + 1) * 2]])
            row, col = self._data[edge_type].edge_index
            edge_weight = torch.ones(len(row))
            adj_temp = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())))
            if adj is None:
                adj = adj_temp
            else:
                adj = adj * adj_temp
        adj = adj.tocoo()
        row, col, data = torch.LongTensor(adj.row), torch.LongTensor(adj.col), torch.FloatTensor(adj.data)
        st_index, ed_index = self.node_types.index(node_type_st), self.node_types.index(node_type_ed)
        if st_index == ed_index:
            for node_type in self.node_types[:st_index]:
                row = row - self._data.num_node[node_type]
                col = col - self._data.num_node[node_type]
        elif st_index < ed_index:
            for node_type in self.node_types[:st_index]:
                row = row - self._data.num_node[node_type]
                col = col - self._data.num_node[node_type]
            for node_type in self.node_types[st_index:ed_index]:
                col = col - self._data.num_node[node_type]
            col = col + self._data.num_node[node_type_st]
        else:
            for node_type in self.node_types[:ed_index]:
                None
                col = col - self._data.num_node[node_type]
                row = row - self._data.num_node[node_type]
            for node_type in self.node_types[ed_index:st_index]:
                row = row - self._data.num_node[node_type]
            row = row + self._data.num_node[node_type_ed]
        if undirected is True:
            data = torch.ones(2 * len(data))
            row, col = to_undirected((row, col))
        adj = csr_matrix((data.numpy(), (row.numpy(), col.numpy())), shape=(num_node, num_node))
        adj.data = torch.ones(len(adj.data)).numpy()
        return adj, feature.numpy(), torch.LongTensor(node_id)

    def nars_preprocess(self, edge_types, predict_class, random_subgraph_num, subgraph_edge_type_num):
        if not isinstance(edge_types, (str, list, tuple)):
            raise TypeError('The given edge types must be a string or a list or a tuple!')
        elif isinstance(edge_types, str):
            edge_types = [edge_types]
        elif isinstance(edge_types, (list, tuple)):
            for edge_type in edge_types:
                if not isinstance(edge_type, str):
                    raise TypeError('Edge type must be a string!')
        adopted_edge_type_combinations = ChooseMultiSubgraphs(subgraph_num=random_subgraph_num, edge_type_num=subgraph_edge_type_num, edge_types=edge_types, predict_class=predict_class)
        if random_subgraph_num > len(adopted_edge_type_combinations):
            random_subgraph_num = len(adopted_edge_type_combinations)
            warnings.warn(f'The input random_subgraph_num exceeds the number of all the combinations of edge types!\nThe random_subgraph_num has been set to {len(adopted_edge_type_combinations)}.', UserWarning)
        chosen_idx = np.random.choice(np.arange(len(adopted_edge_type_combinations)), size=random_subgraph_num, replace=False)
        chosen_edge_types = [tuple(edge_type) for edge_type in np.array(adopted_edge_type_combinations)[chosen_idx]]
        subgraph_dict = {}
        for chosen_edge_type in chosen_edge_types:
            None
            subgraph_dict[chosen_edge_type] = self.sample_by_edge_type(chosen_edge_type)
        return subgraph_dict


class BaseHeteroSGAPModel(nn.Module):

    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseHeteroSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim
        self._pre_graph_op, self._pre_msg_op = None, None
        self._aggregator = None
        self._base_model = None
        self._propagated_feat_list_list = None
        self._processed_feature_list = None
        self._pre_msg_learnable = False

    def preprocess(self, dataset, predict_class, random_subgraph_num=-1, subgraph_edge_type_num=-1, subgraph_list=None):
        if subgraph_list is None and (random_subgraph_num == -1 or subgraph_edge_type_num == -1):
            raise ValueError('Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided!')
        if subgraph_list is not None and (random_subgraph_num != -1 or subgraph_edge_type_num != -1):
            raise ValueError('subgraph_list is provided, random_subgraph_num and subgraph_edge_type_num will be ignored!')
        if not isinstance(dataset, HeteroNodeDataset):
            raise TypeError('Dataset must be an instance of HeteroNodeDataset!')
        elif predict_class not in dataset.node_types:
            raise ValueError('Please input valid node class for prediction!')
        predict_idx = dataset.data.node_id_dict[predict_class]
        if subgraph_list is None:
            subgraph_dict = dataset.nars_preprocess(dataset.edge_types, predict_class, random_subgraph_num, subgraph_edge_type_num)
            subgraph_list = [(key, subgraph_dict[key]) for key in subgraph_dict]
        self._propagated_feat_list_list = [[] for _ in range(self._prop_steps + 1)]
        for key, value in subgraph_list:
            edge_type_list = []
            for edge_type in key:
                edge_type_list.append(edge_type.split('__')[0])
                edge_type_list.append(edge_type.split('__')[2])
            if predict_class in edge_type_list:
                adj, feature, node_id = value
                propagated_feature = self._pre_graph_op.propagate(adj, feature)
                start_pos = list(node_id).index(predict_idx[0])
                for i, feature in enumerate(propagated_feature):
                    self._propagated_feat_list_list[i].append(feature[start_pos:start_pos + dataset.data.num_node[predict_class]])

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        feat_input = []
        for x_list in self._propagated_feat_list_list:
            feat_input.append([])
            for x in x_list:
                feat_input[-1].append(x[idx])
        aggregated_feat_list = self._aggregator(feat_input)
        combined_feat = self._pre_msg_op.aggregate(aggregated_feat_list)
        output = self._base_model(combined_feat)
        return output


class FastBaseHeteroSGAPModel(nn.Module):

    def __init__(self, prop_steps, feat_dim, output_dim):
        super(FastBaseHeteroSGAPModel, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim
        self._pre_graph_op = None
        self._aggregator = None
        self._base_model = None
        self._propagated_feat_list_list = None
        self._processed_feature_list = None
        self._pre_msg_learnable = False

    def preprocess(self, dataset, predict_class, random_subgraph_num=-1, subgraph_edge_type_num=-1, subgraph_list=None):
        if subgraph_list is None and (random_subgraph_num == -1 or subgraph_edge_type_num == -1):
            raise ValueError('Either subgraph_list or (random_subgraph_num, subgraph_edge_type_num) should be provided!')
        if subgraph_list is not None and (random_subgraph_num != -1 or subgraph_edge_type_num != -1):
            raise ValueError('subgraph_list is provided, random_subgraph_num and subgraph_edge_type_num will be ignored!')
        if not isinstance(dataset, HeteroNodeDataset):
            raise TypeError('Dataset must be an instance of HeteroNodeDataset!')
        elif predict_class not in dataset.node_types:
            raise ValueError('Please input valid node class for prediction!')
        predict_idx = dataset.data.node_id_dict[predict_class]
        if subgraph_list is None:
            subgraph_dict = dataset.nars_preprocess(dataset.edge_types, predict_class, random_subgraph_num, subgraph_edge_type_num)
            subgraph_list = [(key, subgraph_dict[key]) for key in subgraph_dict]
        self._propagated_feat_list_list = [[] for _ in range(self._prop_steps + 1)]
        for key, value in subgraph_list:
            edge_type_list = []
            for edge_type in key:
                edge_type_list.append(edge_type.split('__')[0])
                edge_type_list.append(edge_type.split('__')[2])
            if predict_class in edge_type_list:
                adj, feature, node_id = value
                propagated_feature = self._pre_graph_op.propagate(adj, feature)
                start_pos = list(node_id).index(predict_idx[0])
                for i, feature in enumerate(propagated_feature):
                    self._propagated_feat_list_list[i].append(feature[start_pos:start_pos + dataset.data.num_node[predict_class]])
        self._propagated_feat_list_list = [torch.stack(x, dim=2) for x in self._propagated_feat_list_list]
        self._propagated_feat_list_list = torch.stack(self._propagated_feat_list_list, dim=3)
        shape = self._propagated_feat_list_list.size()
        self._propagated_feat_list_list = self._propagated_feat_list_list.view(shape[0], shape[1], shape[2] * shape[3])

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        feat_input = self._propagated_feat_list_list[idx]
        aggregated_feat_from_diff_hops = self._aggregator(feat_input)
        output = self._base_model(aggregated_feat_from_diff_hops)
        return output


class BaseSGAPModelDist(nn.Module):

    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGAPModelDist, self).__init__()
        self._prop_steps = prop_steps
        self._feat_dim = feat_dim
        self._output_dim = output_dim
        self._pre_graph_op, self._pre_msg_op = None, None
        self._post_graph_op, self._post_msg_op = None, None
        self._base_model = None
        self._processed_feat_list = None
        self._processed_feature = None
        self._pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self._pre_graph_op is not None:
            self._processed_feat_list = self._pre_graph_op.propagate(adj, feature)
        else:
            self._processed_feat_list = [feature]

    def postprocess(self, adj, output):
        if self._post_graph_op is not None:
            if self._post_msg_op.aggr_type in ['proj_concat', 'learnable_weighted', 'iterate_learnable_weighted']:
                raise ValueError('Learnable weighted message operator is not supported in the post-processing phase!')
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self._post_graph_op.propagate(adj, output)
            output = self._post_msg_op.aggregate(output)
        return output

    def model_forward(self, idx, device):
        return self.evaluate_forward(idx, device)

    def evaluate_forward(self, idx, device):
        transferred_feat_list = [feat[idx] for feat in self._processed_feat_list]
        processed_feature = self._pre_msg_op.aggregate(transferred_feat_list)
        output = self._base_model(processed_feature)
        return output

    def forward(self, transferred_feat_list):
        processed_feature = self._pre_msg_op.aggregate(transferred_feat_list)
        output = self._base_model(processed_feature)
        return output


class OneDimConvolution(nn.Module):

    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(OneDimConvolution, self).__init__()
        self.__hop_num = prop_steps
        self.__learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.__learnable_weight.append(nn.Parameter(torch.FloatTensor(feat_dim, num_subgraphs)))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.__learnable_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (adopted_feat * self.__learnable_weight[i].unsqueeze(dim=0)).mean(dim=2)
            aggregated_feat_list.append(intermediate_feat)
        return aggregated_feat_list


class OneDimConvolutionWeightSharedAcrossFeatures(nn.Module):

    def __init__(self, num_subgraphs, prop_steps):
        super(OneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
        self.__hop_num = prop_steps
        self.__learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.__learnable_weight.append(nn.Parameter(torch.FloatTensor(1, num_subgraphs)))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.__learnable_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (adopted_feat * self.__learnable_weight[i]).mean(dim=2)
            aggregated_feat_list.append(intermediate_feat)
        return aggregated_feat_list


class FastOneDimConvolution(nn.Module):

    def __init__(self, num_subgraphs, prop_steps):
        super(FastOneDimConvolution, self).__init__()
        self.__num_subgraphs = num_subgraphs
        self.__prop_steps = prop_steps
        self.__learnable_weight = nn.Parameter(torch.ones(num_subgraphs * prop_steps, 1))

    def forward(self, feat_list_list):
        return (feat_list_list @ self.__learnable_weight).squeeze(dim=2)

    @property
    def subgraph_weight(self):
        return self.__learnable_weight.view(self.__num_subgraphs, self.__prop_steps).sum(dim=1)


class IdenticalMapping(nn.Module):

    def __init__(self) ->None:
        super(IdenticalMapping, self).__init__()

    def forward(self, feature):
        return feature


class LogisticRegression(nn.Module):

    def __init__(self, feat_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.__fc = nn.Linear(feat_dim, output_dim)

    def forward(self, feature):
        output = self.__fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.5, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError('MLP must have at least two layers!')
        self.__num_layers = num_layers
        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, output_dim))
        self.__bn = bn
        if self.__bn is True:
            self.__bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.__bns.append(nn.BatchNorm1d(hidden_dim))
        self.__dropout = nn.Dropout(dropout)
        self.__prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for fc in self.__fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.__num_layers - 1):
            feature = self.__fcs[i](feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)
            feature = self.__prelu(feature)
            feature = self.__dropout(feature)
        output = self.__fcs[-1](feature)
        return output


class ResMultiLayerPerceptron(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
        super(ResMultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError('ResMLP must have at least two layers!')
        self.__num_layers = num_layers
        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, output_dim))
        self.__bn = bn
        if self.__bn is True:
            self.__bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.__bns.append(nn.BatchNorm1d(hidden_dim))
        self.__dropout = nn.Dropout(dropout)
        self.__relu = nn.ReLU()

    def forward(self, feature):
        feature = self.__dropout(feature)
        feature = self.__fcs[0](feature)
        if self.__bn is True:
            feature = self.__bns[0](feature)
        feature = self.__relu(feature)
        residual = feature
        for i in range(1, self.__num_layers - 1):
            feature = self.__dropout(feature)
            feature = self.__fcs[i](feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)
            feature_ = self.__relu(feature)
            feature = feature_ + residual
            residual = feature_
        feature = self.__dropout(feature)
        output = self.__fcs[-1](feature)
        return output


class MessageOp(nn.Module):

    def __init__(self, start=None, end=None):
        super(MessageOp, self).__init__()
        self._aggr_type = None
        self._start, self._end = start, end

    @property
    def aggr_type(self):
        return self._aggr_type

    def _combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError('The input must be a list consists of feature matrices!')
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError('The feature matrices must be tensors!')
        return self._combine(feat_list)


class ConcatMessageOp(MessageOp):

    def __init__(self, start, end):
        super(ConcatMessageOp, self).__init__(start, end)
        self._aggr_type = 'concat'

    def _combine(self, feat_list):
        return torch.hstack(feat_list[self._start:self._end])


class IterateLearnableWeightedMessageOp(MessageOp):

    def __init__(self, start, end, combination_type, *args):
        super(IterateLearnableWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = 'iterate_learnable_weighted'
        if combination_type not in ['recursive']:
            raise ValueError("Invalid weighted combination type! Type must be 'recursive'.")
        self.__combination_type = combination_type
        self.__learnable_weight = None
        if combination_type == 'recursive':
            if len(args) != 1:
                raise ValueError('Invalid parameter numbers for the recursive iterate weighted aggregator!')
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim + feat_dim, 1)

    def _combine(self, feat_list):
        weight_list = None
        if self.__combination_type == 'recursive':
            weighted_feat = feat_list[self._start]
            for i in range(self._start, self._end):
                weights = torch.sigmoid(self.__learnable_weight(torch.hstack((feat_list[i], weighted_feat))))
                if i == self._start:
                    weight_list = weights
                else:
                    weight_list = torch.hstack((weight_list, weights))
                weight_list = F.softmax(weight_list, dim=1)
                weighted_feat = torch.mul(feat_list[self._start], weight_list[:, 0].view(-1, 1))
                for j in range(1, i + 1):
                    weighted_feat = weighted_feat + torch.mul(feat_list[self._start + j], weight_list[:, j].view(-1, 1))
        else:
            raise NotImplementedError
        return weighted_feat


def one_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError('This function is designed for list(feature) and tensor(weight)!')
    elif len(feat_list) != weight_list.shape[0]:
        raise ValueError('The feature list and the weight list have different lengths!')
    elif len(weight_list.shape) != 1:
        raise ValueError('The weight list should be a 1d tensor!')
    feat_shape = feat_list[0].shape
    feat_reshape = torch.vstack([feat.view(1, -1).squeeze(0) for feat in feat_list])
    weighted_feat = (feat_reshape * weight_list.view(-1, 1)).sum(dim=0).view(feat_shape)
    return weighted_feat


def two_dim_weighted_add(feat_list, weight_list):
    if not isinstance(feat_list, list) or not isinstance(weight_list, Tensor):
        raise TypeError('This function is designed for list(feature) and tensor(weight)!')
    elif len(feat_list) != weight_list.shape[1]:
        raise ValueError('The feature list and the weight list have different lengths!')
    elif len(weight_list.shape) != 2:
        raise ValueError('The weight list should be a 2d tensor!')
    feat_reshape = torch.stack(feat_list, dim=2)
    weight_reshape = weight_list.unsqueeze(dim=2)
    weighted_feat = torch.bmm(feat_reshape, weight_reshape).squeeze(dim=2)
    return weighted_feat


class LearnableWeightedMessageOp(MessageOp):

    def __init__(self, start, end, combination_type, *args):
        super(LearnableWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = 'learnable_weighted'
        if combination_type not in ['simple', 'simple_allow_neg', 'gate', 'ori_ref', 'jk']:
            raise ValueError("Invalid weighted combination type! Type must be 'simple', 'simple_allow_neg', 'gate', 'ori_ref' or 'jk'.")
        self.__combination_type = combination_type
        self.__learnable_weight = None
        if combination_type == 'simple' or combination_type == 'simple_allow_neg':
            if len(args) != 1:
                raise ValueError('Invalid parameter numbers for the simple learnable weighted aggregator!')
            prop_steps = args[0]
            tmp_2d_tensor = torch.FloatTensor(1, prop_steps + 1)
            nn.init.xavier_normal_(tmp_2d_tensor)
            self.__learnable_weight = Parameter(tmp_2d_tensor.view(-1))
        elif combination_type == 'gate':
            if len(args) != 1:
                raise ValueError('Invalid parameter numbers for the gate learnable weighted aggregator!')
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim, 1)
        elif combination_type == 'ori_ref':
            if len(args) != 1:
                raise ValueError('Invalid parameter numbers for the ori_ref learnable weighted aggregator!')
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim + feat_dim, 1)
        elif combination_type == 'jk':
            if len(args) != 2:
                raise ValueError('Invalid parameter numbers for the jk learnable weighted aggregator!')
            prop_steps, feat_dim = args[0], args[1]
            self.__learnable_weight = Linear(feat_dim + (prop_steps + 1) * feat_dim, 1)

    def _combine(self, feat_list):
        weight_list = None
        if self.__combination_type == 'simple':
            weight_list = F.softmax(torch.sigmoid(self.__learnable_weight[self._start:self._end]), dim=0)
        elif self.__combination_type == 'simple_allow_neg':
            weight_list = self.__learnable_weight[self._start:self._end]
        elif self.__combination_type == 'gate':
            adopted_feat_list = torch.vstack(feat_list[self._start:self._end])
            weight_list = F.softmax(torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(self._end - self._start, -1).T), dim=1)
        elif self.__combination_type == 'ori_ref':
            reference_feat = feat_list[0].repeat(self._end - self._start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(feat_list[self._start:self._end])))
            weight_list = F.softmax(torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(-1, self._end - self._start)), dim=1)
        elif self.__combination_type == 'jk':
            reference_feat = torch.hstack(feat_list).repeat(self._end - self._start, 1)
            adopted_feat_list = torch.hstack((reference_feat, torch.vstack(feat_list[self._start:self._end])))
            weight_list = F.softmax(torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(-1, self._end - self._start)), dim=1)
        else:
            raise NotImplementedError
        weighted_feat = None
        if self.__combination_type == 'simple' or self.__combination_type == 'simple_allow_neg':
            weighted_feat = one_dim_weighted_add(feat_list[self._start:self._end], weight_list=weight_list)
        elif self.__combination_type in ['gate', 'ori_ref', 'jk']:
            weighted_feat = two_dim_weighted_add(feat_list[self._start:self._end], weight_list=weight_list)
        else:
            raise NotImplementedError
        return weighted_feat


class MaxMessageOp(MessageOp):

    def __init__(self, start, end):
        super(MaxMessageOp, self).__init__(start, end)
        self._aggr_type = 'max'

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).max(dim=0)[0]


class MinMessageOp(MessageOp):

    def __init__(self, start, end):
        super(MinMessageOp, self).__init__(start, end)
        self._aggr_type = 'min'

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).min(dim=0)[0]


class OverSmoothDistanceWeightedOp(MessageOp):

    def __init__(self):
        super(OverSmoothDistanceWeightedOp, self).__init__()
        self._aggr_type = 'over_smooth_dis_weighted'

    def _combine(self, feat_list):
        weight_list = []
        features = feat_list[0]
        norm_fea = torch.norm(features, 2, 1).add(1e-10)
        for fea in feat_list:
            norm_cur = torch.norm(fea, 2, 1).add(1e-10)
            tmp = torch.div((features * fea).sum(1), norm_cur)
            tmp = torch.div(tmp, norm_fea)
            weight_list.append(tmp.unsqueeze(-1))
        weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)
        hops = len(feat_list)
        num_nodes = features.shape[0]
        output = []
        for i in range(num_nodes):
            fea = 0.0
            for j in range(hops):
                fea += (weight[i][j] * feat_list[j][i]).unsqueeze(0)
            output.append(fea)
        output = torch.cat(output, dim=0)
        return output


class ProjectedConcatMessageOp(MessageOp):

    def __init__(self, start, end, feat_dim, hidden_dim, num_layers):
        super(ProjectedConcatMessageOp, self).__init__(start, end)
        self._aggr_type = 'proj_concat'
        self.__learnable_weight = ModuleList()
        for _ in range(end - start):
            self.__learnable_weight.append(MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, hidden_dim))

    def _combine(self, feat_list):
        adopted_feat_list = feat_list[self._start:self._end]
        concat_feat = self.__learnable_weight[0](adopted_feat_list[0])
        for i in range(1, self._end - self._start):
            transformed_feat = F.relu(self.__learnable_weight[i](adopted_feat_list[i]))
            concat_feat = torch.hstack((concat_feat, transformed_feat))
        return concat_feat


class SimpleWeightedMessageOp(MessageOp):

    def __init__(self, start, end, combination_type, *args):
        super(SimpleWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = 'simple_weighted'
        if combination_type not in ['alpha', 'hand_crafted']:
            raise ValueError("Invalid weighted combination type! Type must be 'alpha' or 'hand_crafted'.")
        self.__combination_type = combination_type
        if len(args) != 1:
            raise ValueError('Invalid parameter numbers for the simple weighted aggregator!')
        self.__alpha, self.__weight_list = None, None
        if combination_type == 'alpha':
            self.__alpha = args[0]
            if not isinstance(self.__alpha, float):
                raise TypeError('The alpha must be a float!')
            elif self.__alpha > 1 or self.__alpha < 0:
                raise ValueError('The alpha must be a float in [0,1]!')
        elif combination_type == 'hand_crafted':
            self.__weight_list = args[0]
            if isinstance(self.__weight_list, list):
                self.__weight_list = torch.FloatTensor(self.__weight_list)
            elif not isinstance(self.__weight_list, (list, Tensor)):
                raise TypeError('The input weight list must be a list or a tensor!')

    def _combine(self, feat_list):
        if self.__combination_type == 'alpha':
            self.__weight_list = [self.__alpha]
            for _ in range(len(feat_list) - 1):
                self.__weight_list.append((1 - self.__alpha) * self.__weight_list[-1])
            self.__weight_list = torch.FloatTensor(self.__weight_list[self._start:self._end])
        elif self.__combination_type == 'hand_crafted':
            pass
        else:
            raise NotImplementedError
        weighted_feat = one_dim_weighted_add(feat_list[self._start:self._end], weight_list=self.__weight_list)
        return weighted_feat


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IdenticalMapping,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogisticRegression,
     lambda: ([], {'feat_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_PKU_DAIR_SGL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

