import sys
_module = sys.modules[__name__]
del sys
master = _module
check_style = _module
tensorboard_pytorch = _module
tensorboard_tf = _module
util = _module
vis_pred = _module
visualize = _module
ml3d = _module
configs = _module
datasets = _module
argoverse = _module
augment = _module
augmentation = _module
base_dataset = _module
customdataset = _module
inference_dummy = _module
kitti = _module
lyft = _module
matterport_objects = _module
nuscenes = _module
parislille3d = _module
s3dis = _module
samplers = _module
semseg_random = _module
semseg_spatially_regular = _module
scannet = _module
semantic3d = _module
semantickitti = _module
shapenet = _module
sunrgbd = _module
toronto3d = _module
utils = _module
bev_box = _module
dataprocessing = _module
operations = _module
transforms = _module
waymo = _module
metrics = _module
mAP = _module
tf = _module
dataloaders = _module
tf_dataloader = _module
models = _module
base_model = _module
base_model_objdet = _module
kpconv = _module
network_blocks = _module
openvino_model = _module
point_pillars = _module
point_rcnn = _module
point_transformer = _module
pvcnn = _module
randlanet = _module
sparseconvnet = _module
kernels = _module
kernel_points = _module
min = _module
modules = _module
losses = _module
cross_entropy = _module
focal_loss = _module
semseg_loss = _module
smooth_L1 = _module
semseg_metric = _module
optimizers = _module
pointnet = _module
schedulers = _module
bn_momentum_scheduler = _module
cosine_warmup_scheduler = _module
lr_one_cycle_scheduler = _module
pipelines = _module
base_pipeline = _module
object_detection = _module
semantic_segmentation = _module
objdet_helper = _module
pointnet2_modules = _module
pointnet2_utils = _module
tf_utils = _module
roipool3d_utils = _module
concat_batcher = _module
default_batcher = _module
torch_dataloader = _module
torch_sampler = _module
models = _module
base_model = _module
base_model_objdet = _module
kpconv = _module
openvino_model = _module
point_pillars = _module
point_rcnn = _module
point_transformer = _module
pvcnn = _module
randlanet = _module
sparseconvnet = _module
modules = _module
cross_entropy = _module
focal_loss = _module
semseg_loss = _module
smooth_L1 = _module
semseg_metric = _module
optim_wrapper = _module
pointnet = _module
bn_momentum_scheduler = _module
cosine_warmup_scheduler = _module
pipelines = _module
base_pipeline = _module
object_detection = _module
semantic_segmentation = _module
utils = _module
objdet_helper = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
roipool3d = _module
roipool3d_utils = _module
torch_utils = _module
builder = _module
config = _module
dataset_helper = _module
log = _module
registry = _module
vis = _module
boundingbox = _module
colormap = _module
labellut = _module
visualizer = _module
collect_bboxes = _module
demo_api_train = _module
demo_datasets = _module
demo_obj_det = _module
preprocess_argoverse = _module
preprocess_lyft = _module
preprocess_nuscenes = _module
preprocess_scannet = _module
preprocess_semantic3d = _module
preprocess_sunrgbd = _module
preprocess_waymo = _module
run_pipeline = _module
setup = _module
test_integration = _module
test_models = _module

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


import numpy as np


from torch.utils.tensorboard import SummaryWriter


import logging


import tensorflow as tf


import random


import functools


from tensorflow.python.framework import ops


from functools import partial


import time


import torch


import math


from torch.utils.data import Sampler


from torch.utils.data import get_worker_info


import re


import collections


from torch.utils.data import Dataset


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from torch.nn.parameter import Parameter


from torch.nn.init import kaiming_uniform_


from sklearn.neighbors import KDTree


import matplotlib.pyplot as plt


from matplotlib import cm


import copy


from torch import nn


from torch.nn import functional as F


from torch.nn.modules.utils import _pair


import torch.utils.dlpack


from torch.autograd import Function


import torch.nn.functional as F


from torch.autograd import Variable


import warnings


from collections.abc import Iterable


from typing import List


import torch.optim.lr_scheduler as lr_sched


import torch.distributed as dist


from torch.utils.data import DataLoader


from typing import Tuple


import inspect


from collections import deque


from torch import multiprocessing


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


class Config(object):

    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict should be a dict, butgot {type(cfg_dict)}')
        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        self.cfg_dict = cfg_dict

    def dump(self, *args, **kwargs):
        """Dump to a string."""

        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, ConfigDict):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict
        self_as_dict = convert_to_dict(self._cfg_dict, [])
        None
        return yaml.dump(self_as_dict, *args, **kwargs)

    def convert_to_tf_names(self, name):
        """Convert keys compatible with tensorflow."""
        cfg = self._cfg_dict
        with open(os.path.join(Path(__file__).parent, '../configs/torch_to_tf.yml')) as f:
            mapping = yaml.safe_load(f)[name]

        def convert_dict(cfg, mapping):
            cfg_new = {}
            for key in cfg:
                if isinstance(cfg[key], dict):
                    cfg_new[key] = convert_dict(cfg[key], mapping)
                elif key in mapping:
                    item = cfg[key]
                    if isinstance(mapping[key], list):
                        for k, v in zip(mapping[key], item):
                            cfg_new[k] = v
                    else:
                        cfg_new[mapping[key]] = item
                else:
                    cfg_new[key] = cfg[key]
            return cfg_new
        cfg = convert_dict(cfg, mapping)
        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg))

    @staticmethod
    def merge_cfg_file(cfg, args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        if args.device is not None:
            cfg.pipeline.device = args.device
            cfg.model.device = args.device
        if args.split is not None:
            cfg.pipeline.split = args.split
        if args.main_log_dir is not None:
            cfg.pipeline.main_log_dir = args.main_log_dir
        if args.dataset_path is not None:
            cfg.dataset.dataset_path = args.dataset_path
        if args.ckpt_path is not None:
            cfg.model.ckpt_path = args.ckpt_path
        extra_cfg_dict = {'model': {}, 'dataset': {}, 'pipeline': {}}
        for full_key, v in extra_dict.items():
            d = extra_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v
        cfg_dict_dataset = Config._merge_a_into_b(extra_cfg_dict['dataset'], cfg.dataset)
        cfg_dict_pipeline = Config._merge_a_into_b(extra_cfg_dict['pipeline'], cfg.pipeline)
        cfg_dict_model = Config._merge_a_into_b(extra_cfg_dict['model'], cfg.model)
        return cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model

    @staticmethod
    def merge_module_cfg_file(args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        cfg_dataset = Config.load_from_file(args.cfg_dataset)
        cfg_model = Config.load_from_file(args.cfg_model)
        cfg_pipeline = Config.load_from_file(args.cfg_pipeline)
        cfg_dict = {'dataset': cfg_dataset.cfg_dict, 'model': cfg_model.cfg_dict, 'pipeline': cfg_pipeline.cfg_dict}
        cfg = Config(cfg_dict)
        return Config.merge_cfg_file(cfg, args, extra_dict)

    @staticmethod
    def _merge_a_into_b(a, b):
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict):
                if k in b and not isinstance(b[k], dict):
                    raise TypeError('{}={} in child config cannot inherit from base '.format(k, v) + 'because {} is a dict in the child config but is of '.format(k) + 'type {} in base config.  '.format(type(b[k])))
                b[k] = Config._merge_a_into_b(v, b.get(k, ConfigDict()))
            else:
                if v is None:
                    continue
                if v.isnumeric():
                    v = int(v)
                elif v.replace('.', '').isnumeric():
                    v = float(v)
                elif v == 'True' or v == 'true':
                    v = True
                elif v == 'False' or v == 'false':
                    v = False
                b[k] = v
        return b

    def merge_from_dict(self, new_dict):
        """Merge a new dict into cfg_dict.

        Args:
            new_dict (dict): a dict of configs.
        """
        b = self.copy()
        for k, v in new_dict.items():
            if v is None:
                continue
            b[k] = v
        return Config(b)

    @staticmethod
    def load_from_file(filename):
        if filename is None:
            return Config()
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
                temp_config_name = os.path.basename(temp_config_file.name)
                shutil.copyfile(filename, os.path.join(temp_config_dir, temp_config_name))
                temp_module_name = os.path.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith('__')}
                del sys.modules[temp_module_name]
                temp_config_file.close()
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)
        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __getstate__(self):
        return self.cfg_dict

    def __setstate__(self, state):
        self.cfg_dict = state


class BaseModel(ABC, torch.nn.Module):
    """Base dataset class."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            cfg (cfg object or str): cfg object or path to cfg file
            dataset_path (str): Path to the dataset
            **kwargs (dict): Dict of args
        """
        super().__init__()
        self.cfg = Config(kwargs)
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    @abstractmethod
    def get_loss(self, results, inputs):
        """Computes the loss given the network input and outputs.

        Args:
            Loss: A loss object.
            results: This is the output of the model.
            inputs: This is the input to the model.

        Returns:
            Returns the loss value.
        """
        return

    @abstractmethod
    def get_optimizer(self, cfg_pipeline):
        """Returns an optimizer object for the model.

        Args:
            cfg_pipeline: A Config object with the configuration of the pipeline.

        Returns:
            Returns a new optimizer object.
        """
        return

    @abstractmethod
    def preprocess(self, cfg_pipeline):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data
        """
        return

    @abstractmethod
    def transform(self, cfg_pipeline):
        """Transform function for the point cloud and features.

        Args:
            cfg_pipeline: config file for pipeline.
        """
        return

    @abstractmethod
    def inference_end(self, results, attr=None):
        """This function is called after the inference.

        This function can be implemented to apply post-processing on the
        network outputs.

        Args:
            results: The model outputs as returned by the call() function.
                Post-processing is applied on this object.

        Returns:
            Returns True if the inference is complete and otherwise False.
            Returning False can be used to implement inference for large point
            clouds which require multiple passes.
        """
        return


class DataProcessing:

    @staticmethod
    def grid_subsampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """CPP wrapper for a grid subsampling (method = barycenter for points and
        features).

        Args:
            points: (N, 3) matrix of input points
            features: optional (N, d) matrix of features (floating number)
            labels: optional (N,) matrix of integer labels
            grid_size: parameter defining the size of grid voxels
            verbose: 1 to display

        Returns:
            Subsampled points, with features and/or labels depending of the input
        """
        if features is None and labels is None:
            return subsample(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return subsample(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return subsample(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return subsample(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)

    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape(-1)
        sem_label = label & 65535
        inst_label = label >> 16
        assert (sem_label + (inst_label << 16) == label).all()
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """KNN search.

        Args:
            support_pts: points you have, N1*3
            query_pts: points you want to know the neighbour index, N2*3
            k: Number of neighbours in knn search

        Returns:
            neighbor_idx: neighboring points indexes, N2*k
        """
        nns = o3c.nns.NearestNeighborSearch(o3c.Tensor.from_numpy(support_pts))
        nns.knn_index()
        idx, dist = nns.knn_search(o3c.Tensor.from_numpy(query_pts), k)
        return idx.numpy().astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def IoU_from_confusions(confusions):
        """Computes IoU from confusion matrices.

        Args:
            confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes

        Returns:
            ([..., n_c] np.float32) IoU score
        """
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-06)
        mask = TP_plus_FN < 0.001
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-06)
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def Acc_from_confusions(confusions):
        return confusions.diagonal() / confusions.sum(axis=0)

    @staticmethod
    def get_class_weights(num_per_class):
        num_per_class = np.array(num_per_class, dtype=np.float32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
    def invT(T):
        R = T[:3, :3]
        t = T[3:, :3]
        R = np.linalg.inv(R)
        t = t @ -R
        M = np.concatenate([R, t], axis=0)
        return np.concatenate([M, [[0], [0], [0], [1]]], axis=1)

    @staticmethod
    def world2cam(points, world_cam):
        points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
        for i in range(len(points) // 10000 + 1):
            points[i * 10000:(i + 1) * 10000] = np.matmul(points[i * 10000:(i + 1) * 10000], world_cam)
        return points[..., :3]

    @staticmethod
    def cam2img(points, cam_img):
        points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
        for i in range(len(points) // 10000 + 1):
            points[i * 10000:(i + 1) * 10000] = np.matmul(points[i * 10000:(i + 1) * 10000], cam_img)
        pts_img = (points[:, :2].T / points[:, 3]).T
        depth = points[:, 2] - cam_img[3, 2]
        return pts_img, depth

    @staticmethod
    def cam2world(points, world_cam):
        cam_world = DataProcessing.invT(world_cam)
        points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
        return np.matmul(points, cam_world)[..., :3]

    @staticmethod
    def remove_outside_points(points, world_cam, cam_img, image_shape):
        """Remove points which are outside of image.

        Args:
            points (np.ndarray, shape=[N, 3+dims]): Total points.
            world_cam (np.ndarray, shape=[4, 4]): Matrix to project points in
                lidar coordinates to camera coordinates.
            cam_img (p.array, shape=[4, 4]): Matrix to project points in
                camera coordinates to image coordinates.
            image_shape (list[int]): Shape of image.

        Returns:
            np.ndarray, shape=[N, 3+dims]: Filtered points.
        """
        pts_cam = DataProcessing.world2cam(points[:, :3], world_cam)
        pts_img, depth = DataProcessing.cam2img(pts_cam, cam_img)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < image_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < image_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        valid = np.logical_and(val_flag_merge, depth >= 0)
        return points[valid]


class BatchNormBlock(nn.Module):

    def __init__(self, m, eps=0.0001, momentum=0.01):
        super(BatchNormBlock, self).__init__()
        self.bn = nn.BatchNorm1d(m, eps=eps, momentum=momentum)

    def forward(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.bn(torch.cat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l
        return out_list

    def __name__(self):
        return 'BatchNormBlock'


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False, l_relu=0.1):
        """Initialize a standard unary block with its ReLU and BatchNorm.

        Args:
            in_dim: dimension input features
            out_dim: dimension input features
            use_bn: boolean indicating if we use Batch Norm
            bn_momentum: Batch norm momentum
            no_relu: Do not use leaky ReLU
            l_relu: Leaky ReLU factor
        """
        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(l_relu)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim, self.out_dim, str(self.use_bn), str(not self.no_relu))


def global_average(x, batch_lengths):
    """Block performing a global average over batch pooling.

    Args:
        x: [N, D] input features
        batch_lengths: [B] list of batch lengths

    Returns:
        [B, D] averaged features
    """
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))
        i0 += length
    return torch.stack(averaged_features)


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """Initialize a global average block with its ReLU and BatchNorm."""
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])


def gather(x, idx, method=2):
    """Implementation of a custom gather operation for faster backwards.

    Args:
        x: Input with shape [N, D_1, ... D_d]
        idx: Indexing with shape [n_1, ..., n_m]
        method: Choice of the method

    Returns:
        x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def max_pool(x, inds):
    """Pools features with the maximum values.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] pooling indices

    Returns:
        [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    pool_features = gather(x, inds)
    max_features, _ = torch.max(pool_features, 1)
    return max_features


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """Initialize a max pooling block with its ReLU and BatchNorm."""
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


def closest_pool(x, inds):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] Only the first column is used for pooling

    Returns:
        [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    return gather(x, inds[:, 0])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """Initialize a nearest upsampling block with its ReLU and BatchNorm."""
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind, self.layer_ind - 1)


def create_3D_rotations(axis, angle):
    """Create rotation matrices from a list of axes and angles. Code from
    wikipedia on quaternions.

    Args:
        axis: float32[N, 3]
        angle: float32[N,]

    Returns:
        float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20, t11 - t12, t19 + t20, t1 + t2 * t24], axis=1)
    return np.reshape(R, (-1, 3, 3))


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', ratio=0.66, verbose=0):
    """Creation of kernel point via optimization of potentials.

    Args:
        radius: Radius of the kernels
        num_points: points composing kernels
        num_kernels: number of wanted kernels
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        ratio: ratio of the radius where you want the kernels points to be placed
        verbose: display option

    Returns:
        points [num_kernels, num_points, dimension]
    """
    radius0 = 1
    diameter0 = 2
    moving_factor = 0.01
    continuous_moving_decay = 0.9995
    thresh = 1e-05
    clip = 0.05 * radius0
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-06)
        inter_grads = np.sum(inter_grads, axis=1)
        circle_grads = 10 * kernel_points
        gradients = inter_grads + circle_grads
        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-06, -1)
        if verbose:
            None
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            None
        moving_factor *= continuous_moving_decay
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])
    return kernel_points * radius, saved_gradient_norms


class bcolors:
    WARNING = '\x1b[93m'
    ENDC = '\x1b[0m'


def spherical_Lloyd(radius, num_cells, dimension=3, fixed='center', approximation='monte-carlo', approx_n=5000, max_iter=500, momentum=0.9, verbose=0):
    """Creation of kernel point via Lloyd algorithm. We use an approximation of
    the algorithm, and compute the Voronoi cell centers with discretization  of
    space. The exact formula is not trivial with part of the sphere as sides.

    Args:
        radius: Radius of the kernels
        num_cells: Number of cell (kernel points) in the Voronoi diagram.
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
        approx_n: Number of point used for approximation.
        max_iter: Maximum nu;ber of iteration for the algorithm.
        momentum: Momentum of the low pass filter smoothing kernel point positions
        verbose: display option

    Returns:
        points [num_kernels, num_points, dimension]
    """
    radius0 = 1.0
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[np.logical_and(d2 < radius0 ** 2, (0.9 * radius0) ** 2 < d2), :]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))
    if fixed == 'center':
        kernel_points[0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    if approximation == 'discretization':
        side_n = int(np.floor(approx_n ** (1.0 / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError('Unsupported dimension (max is 4)')
    elif approximation == 'monte-carlo':
        X = np.zeros((0, dimension))
    else:
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]
    warning = False
    max_moves = np.zeros((0,))
    for iter in range(max_iter):
        if approximation == 'monte-carlo':
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = cell_inds == c
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))
        if fixed == 'center':
            kernel_points[0, :] *= 0
        if fixed == 'verticals':
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0
        if verbose:
            None
            if warning:
                None
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker='.', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
    if verbose:
        if dimension == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker='.', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect('equal')
            plt.title('Check if kernel is correct.')
            plt.draw()
            plt.show()
        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title('Check if kernel is correct.')
            plt.show()
    return kernel_points * radius


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)
    if num_kpoints > 30:
        lloyd = True
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.npy'.format(num_kpoints, fixed, dimension))
    if not exists(kernel_file):
        if lloyd:
            kernel_points = spherical_Lloyd(1.0, num_kpoints, dimension=dimension, fixed=fixed, verbose=0)
        else:
            kernel_points, grad_norms = kernel_point_optimization_debug(1.0, num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0)
            best_k = np.argmin(grad_norms[-1, :])
            kernel_points = kernel_points[best_k, :, :]
        np.save(kernel_file, kernel_points)
    else:
        kernel_points = np.load(kernel_file)
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
    elif dimension == 3:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        else:
            phi = (np.random.rand() - 0.5) * np.pi
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            alpha = np.random.rand() * 2 * np.pi
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
            R = R.astype(np.float32)
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)
    kernel_points = radius * kernel_points
    kernel_points = np.matmul(kernel_points, R)
    return kernel_points.astype(np.float32)


def radius_gaussian(sq_r, sig, eps=1e-09):
    """Compute a radius gaussian (gaussian of distance)

    Args:
        sq_r: input radiuses [dn, ..., d1, d0]
        sig: extents of gaussians [d1, d0] or [d0] or float

    Returns:
        gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius, fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum', deformable=False, modulated=False):
        """Initialize parameters for KPConvDeformable.

        Args:
            kernel_size: Number of kernel points.
            p_dim: dimension of the point space.
            in_channels: dimension of input features.
            out_channels: dimension of output features.
            KP_extent: influence radius of each kernel point.
            radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
            fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
            KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
            aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
            deformable: choose deformable or not
            modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K, self.p_dim, self.in_channels, self.offset_dim, KP_extent, radius, fixed_kernel_points=fixed_kernel_points, KP_influence=KP_influence, aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None
        self.reset_parameters()
        if deformable:
            self.kernel_points = self.offset_conv.kernel_points
        else:
            self.kernel_points = self.init_KP()
        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """Initialize the kernel point positions in a sphere

        Returns:
            the tensor of kernel points
        """
        K_points_numpy = load_kernels(self.radius, self.K, dimension=self.p_dim, fixed=self.fixed_kernel_points)
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        if self.deformable:
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias
            if self.modulated:
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])
            else:
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)
                modulations = None
            offsets = unscaled_offsets * self.KP_extent
        else:
            offsets = None
            modulations = None
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1000000.0), 0)
        neighbors = s_pts[neighb_inds, :]
        neighbors = neighbors - q_pts.unsqueeze(1)
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points
        sq_distances = torch.sum(differences ** 2, dim=3)
        if self.deformable:
            self.min_d2, _ = torch.min(sq_distances, dim=1)
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds
        if self.KP_influence == 'constant':
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'linear':
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'gaussian':
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)
        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
        neighb_x = gather(x, new_neighb_inds)
        weighted_features = torch.matmul(all_weights, neighb_x)
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius, self.in_channels, self.out_channels)


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """Initialize a resnet bottleneck block.

        Args:
            block_name: Block name
            in_dim: dimension input features
            out_dim: dimension input features
            radius: current radius of convolution
            layer_ind: Layer ind
            config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()
        current_extent = radius * config.KP_extent / config.conv_radius
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        l_relu = config.get('l_relu', 0.1)
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum, l_relu=l_relu)
        else:
            self.unary1 = nn.Identity()
        self.KPConv = KPConv(config.num_kernel_points, config.in_points_dim, out_dim // 4, out_dim // 4, current_extent, radius, fixed_kernel_points=config.fixed_kernel_points, KP_influence=config.KP_influence, aggregation_mode=config.aggregation_mode, deformable='deform' in block_name, modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True, l_relu=l_relu)
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True, l_relu=l_relu)
        else:
            self.unary_shortcut = nn.Identity()
        self.leaky_relu = nn.LeakyReLU(l_relu)
        return

    def forward(self, features, batch):
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
        x = self.unary1(features)
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))
        x = self.unary2(x)
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)
        return self.leaky_relu(x + shortcut)


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """Initialize a simple convolution block with its ReLU and BatchNorm.

        Args:
            block_name: Block name
            in_dim: dimension input features
            out_dim: dimension input features
            radius: current radius of convolution
            layer_ind: Index for layer
            config: parameters
        """
        super(SimpleBlock, self).__init__()
        current_extent = radius * config.KP_extent / config.conv_radius
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.KPConv = KPConv(config.num_kernel_points, config.in_points_dim, in_dim, out_dim // 2, current_extent, radius, fixed_kernel_points=config.fixed_kernel_points, KP_influence=config.KP_influence, aggregation_mode=config.aggregation_mode, deformable='deform' in block_name, modulated=config.modulated)
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(config.get('l_relu', 0.1))
        return

    def forward(self, x, batch):
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.batch_norm(x))


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum, l_relu=config.get('l_relu', 0.1))
    elif block_name in ['simple', 'simple_deformable', 'simple_invariant', 'simple_equivariant', 'simple_strided', 'simple_deformable_strided', 'simple_invariant_strided', 'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif block_name in ['resnetb', 'resnetb_invariant', 'resnetb_equivariant', 'resnetb_deformable', 'resnetb_strided', 'resnetb_deformable_strided', 'resnetb_equivariant_strided', 'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)
    elif block_name == 'global_average':
        return GlobalAverageBlock()
    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)
    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device):
    """Loss functions for semantic segmentation."""
    valid_scores = scores.reshape(-1, num_classes)
    valid_labels = labels.reshape(-1)
    ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
    for ign_label in ignored_label_inds:
        ignored_bool = torch.logical_or(ignored_bool, torch.eq(valid_labels, ign_label))
    valid_idx = torch.where(torch.logical_not(ignored_bool))[0]
    valid_scores = torch.gather(valid_scores, 0, valid_idx.unsqueeze(-1).expand(-1, num_classes))
    valid_labels = torch.gather(valid_labels, 0, valid_idx)
    reducing_list = torch.arange(0, num_classes, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)
    for ign_label in ignored_label_inds:
        if ign_label >= 0:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels.long())
    return valid_scores, valid_labels


def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0
    for m in net.modules():
        if isinstance(m, KPConv) and m.deformable:
            KP_min_d2 = m.min_d2 / m.KP_extent ** 2
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))
            KP_locs = m.deformed_KP / m.KP_extent
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K
    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


def trans_normalize(pc, feat, t_normalize):
    dim = t_normalize.get('recentering', [0, 1, 2])
    pc[:, dim] = pc[:, dim] - pc.mean(0)[dim]
    if t_normalize.get('method', None):
        method = t_normalize['method']
        if method == 'linear':
            if t_normalize.get('normalize_points', False):
                pc -= pc.mean()
                pc /= (pc.max(0) - pc.min(0)).max()
            if feat is not None:
                feat_bias = t_normalize.get('feat_bias', 0)
                feat_scale = t_normalize.get('feat_scale', 1)
                feat -= feat_bias
                feat /= feat_scale
        elif method == 'coords_only':
            feat = None
    return pc, feat


class KPFCNN(BaseModel):
    """Class defining KPFCNN.

    A model for Semantic Segmentation.
    """

    def __init__(self, name='KPFCNN', lbl_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], num_classes=19, ignored_label_inds=[0], ckpt_path=None, batcher='ConcatBatcher', architecture=['simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary'], in_radius=4.0, max_in_points=100000, batch_num=8, batch_limit=30000, val_batch_num=8, num_kernel_points=15, first_subsampling_dl=0.06, conv_radius=2.5, deform_radius=6.0, KP_extent=1.2, KP_influence='linear', aggregation_mode='sum', first_features_dim=128, in_features_dim=2, modulated=False, use_batch_norm=True, batch_norm_momentum=0.02, deform_fitting_mode='point2point', deform_fitting_power=1.0, repulse_extent=1.2, augment_scale_anisotropic=True, augment_symmetries=[True, False, False], augment_rotation='vertical', augment_scale_min=0.8, augment_scale_max=1.2, augment_noise=0.001, augment_color=0.8, in_points_dim=3, fixed_kernel_points='center', num_layers=5, l_relu=0.1, reduce_fc=False, **kwargs):
        super().__init__(name=name, lbl_values=lbl_values, num_classes=num_classes, ignored_label_inds=ignored_label_inds, ckpt_path=ckpt_path, batcher=batcher, architecture=architecture, in_radius=in_radius, max_in_points=max_in_points, batch_num=batch_num, batch_limit=batch_limit, val_batch_num=val_batch_num, num_kernel_points=num_kernel_points, first_subsampling_dl=first_subsampling_dl, conv_radius=conv_radius, deform_radius=deform_radius, KP_extent=KP_extent, KP_influence=KP_influence, aggregation_mode=aggregation_mode, first_features_dim=first_features_dim, in_features_dim=in_features_dim, modulated=modulated, use_batch_norm=use_batch_norm, batch_norm_momentum=batch_norm_momentum, deform_fitting_mode=deform_fitting_mode, deform_fitting_power=deform_fitting_power, repulse_extent=repulse_extent, augment_scale_anisotropic=augment_scale_anisotropic, augment_symmetries=augment_symmetries, augment_rotation=augment_rotation, augment_scale_min=augment_scale_min, augment_scale_max=augment_scale_max, augment_noise=augment_noise, augment_color=augment_color, in_points_dim=in_points_dim, fixed_kernel_points=fixed_kernel_points, num_layers=num_layers, l_relu=l_relu, reduce_fc=reduce_fc, **kwargs)
        cfg = self.cfg
        layer = 0
        r = cfg.first_subsampling_dl * cfg.conv_radius
        in_dim = cfg.in_features_dim
        out_dim = cfg.first_features_dim
        lbl_values = cfg.lbl_values
        ign_lbls = cfg.ignored_label_inds
        self.K = cfg.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.neighborhood_limits = []
        for block_i, block in enumerate(cfg.architecture):
            if 'equivariant' in block and not out_dim % 3 == 0:
                raise ValueError('Equivariant block but features dimension is not a factor of 3')
            if np.any([(tmp in block) for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            if 'upsample' in block:
                break
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, cfg))
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim
            if 'pool' in block or 'strided' in block:
                layer += 1
                r *= 2
                out_dim *= 2
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []
        start_i = 0
        for block_i, block in enumerate(cfg.architecture):
            if 'upsample' in block:
                start_i = block_i
                break
        for block_i, block in enumerate(cfg.architecture[start_i:]):
            if block_i > 0 and 'upsample' in cfg.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, cfg))
            in_dim = out_dim
            if block_i == 0 and cfg.reduce_fc:
                out_dim = out_dim // 2
            if 'upsample' in block:
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        if reduce_fc:
            self.head_mlp = UnaryBlock(out_dim, cfg.first_features_dim // 2, True, cfg.batch_norm_momentum, l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim // 2, self.C, False, 1, no_relu=True, l_relu=cfg.get('l_relu', 0.1))
        else:
            self.head_mlp = UnaryBlock(out_dim, cfg.first_features_dim, False, 0, l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim, self.C, False, 0, l_relu=cfg.get('l_relu', 0.1))
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        self.deform_fitting_mode = cfg.deform_fitting_mode
        self.deform_fitting_power = cfg.deform_fitting_power
        self.repulse_extent = cfg.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        return

    def forward(self, batch):
        x = batch.features.clone().detach()
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)
        return x

    def get_optimizer(self, cfg_pipeline):
        deform_params = [v for k, v in self.named_parameters() if 'offset' in k]
        other_params = [v for k, v in self.named_parameters() if 'offset' not in k]
        deform_lr = cfg_pipeline.learning_rate * cfg_pipeline.deform_lr_factor
        optimizer = torch.optim.SGD([{'params': other_params}, {'params': deform_params, 'lr': deform_lr}], lr=cfg_pipeline.learning_rate, momentum=cfg_pipeline.momentum, weight_decay=cfg_pipeline.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Runs the loss on outputs of the model.

        Args:
            outputs: logits
            labels: labels

        Returns:
            loss
        """
        cfg = self.cfg
        labels = inputs['data'].labels
        outputs = results
        outputs = torch.transpose(results, 0, 1).unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores, labels = filter_valid_label(results, labels, cfg.num_classes, cfg.ignored_label_inds, device)
        self.output_loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)
        loss = self.output_loss + self.reg_loss
        return loss, labels, scores

    def preprocess(self, data, attr):
        cfg = self.cfg
        points = np.array(data['point'][:, 0:3], dtype=np.float32)
        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)
        split = attr['split']
        data = dict()
        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(points, labels=labels, grid_size=cfg.first_subsampling_dl)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(points, features=feat, labels=labels, grid_size=cfg.first_subsampling_dl)
        search_tree = KDTree(sub_points)
        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree
        if split in ['test', 'testing', 'validation', 'valid']:
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds
        return data

    def transform(self, data, attr, is_test=False):
        points = data['point']
        sem_labels = data['label']
        feat = data['feat']
        search_tree = data['search_tree']
        dim_points = points.shape[1]
        if feat is None:
            dim_features = dim_points
        else:
            dim_features = feat.shape[1] + dim_points
        merged_points = np.zeros((0, dim_points), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int32)
        merged_coords = np.zeros((0, dim_features), dtype=np.float32)
        p_origin = np.zeros((1, 4))
        p_origin[0, 3] = 1
        p0 = p_origin[:, :3]
        p0 = np.squeeze(p0)
        o_pts = None
        o_labels = None
        num_merged = 0
        result_data = {'p_list': [], 'f_list': [], 'l_list': [], 'p0_list': [], 's_list': [], 'R_list': [], 'r_inds_list': [], 'r_mask_list': [], 'val_labels_list': [], 'cfg': self.cfg}
        curr_num_points = 0
        max_num_points = min(self.cfg.batch_limit, self.cfg.max_in_points)
        min_in_points = self.cfg.get('min_in_points', 3)
        min_in_points = min(min_in_points, self.cfg.max_in_points)
        while curr_num_points < min_in_points:
            new_points = points.copy()
            curr_new_points, mask_inds, p0 = self.trans_point_sampler(pc=new_points, feat=feat, label=sem_labels, search_tree=search_tree, num_points=min_in_points, radius=self.cfg.in_radius)
            curr_sem_labels = sem_labels[mask_inds]
            o_labels = sem_labels.astype(np.int32)
            curr_new_points = curr_new_points - p0
            t_normalize = self.cfg.get('t_normalize', {})
            curr_new_points, curr_feat = trans_normalize(curr_new_points, feat, t_normalize)
            if curr_feat is None:
                curr_new_coords = curr_new_points.copy()
            else:
                curr_new_coords = np.hstack((curr_new_points, curr_feat[mask_inds, :]))
            in_pts = curr_new_points
            in_fts = curr_new_coords
            in_lbls = curr_sem_labels
            n = in_pts.shape[0]
            residual_num_points = max_num_points - curr_num_points
            if n > residual_num_points:
                input_inds = np.random.choice(n, size=residual_num_points, replace=False)
                in_pts = in_pts[input_inds, :]
                in_fts = in_fts[input_inds, :]
                in_lbls = in_lbls[input_inds]
                mask_inds = mask_inds[input_inds]
                n = input_inds.shape[0]
            curr_num_points += n
            reproj_mask = mask_inds
            if attr['split'] in ['test']:
                proj_inds = data['proj_inds']
            else:
                proj_inds = np.zeros((0,))
            in_pts, scale, R = self.augmentation_transform(in_pts, is_test=is_test)
            if np.random.rand() > self.cfg.augment_color:
                in_fts[:, 3:] *= 0
            result_data['p_list'] += [in_pts]
            result_data['f_list'] += [in_fts]
            result_data['l_list'] += [np.squeeze(in_lbls)]
            result_data['p0_list'] += [p0]
            result_data['s_list'] += [scale]
            result_data['R_list'] += [R]
            result_data['r_inds_list'] += [proj_inds]
            result_data['r_mask_list'] += [reproj_mask]
            result_data['val_labels_list'] += [o_labels]
        return result_data

    def inference_begin(self, data):
        self.test_smooth = 0.98
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = np.random.rand(num_points) * 0.001
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes], dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        self.batcher = ConcatBatcher(self.device)

    def inference_preprocess(self):
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, is_test=True)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs
        return inputs

    def update_probs(self, inputs, results, test_probs, test_labels):
        self.test_smooth = 0.95
        stk_probs = torch.nn.functional.softmax(results, dim=-1)
        stk_probs = stk_probs.cpu().data.numpy()
        batch = inputs['data']
        stk_labels = batch.labels.cpu().data.numpy()
        lengths = batch.lengths[0].cpu().numpy()
        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels
        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            labels = np.argmax(probs, 1)
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            test_probs[proj_mask] = self.test_smooth * test_probs[proj_mask] + (1 - self.test_smooth) * probs
            test_labels[proj_mask] = labels
            i0 += length
        return test_probs, test_labels

    def inference_end(self, inputs, results):
        m_softmax = torch.nn.Softmax(dim=-1)
        stk_probs = m_softmax(results)
        stk_probs = results.cpu().data.numpy()
        batch = inputs['data']
        lengths = batch.lengths[0].cpu().numpy()
        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels
        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            self.test_probs[proj_mask] = self.test_smooth * self.test_probs[proj_mask] + (1 - self.test_smooth) * probs
            i0 += length
        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] - self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)
            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            inference_result = {'predict_labels': pred_labels, 'predict_scores': test_probs}
            data = self.inference_ori_data
            acc = (pred_labels == data['label'] - 1).mean()
            self.inference_result = inference_result
            return True
        else:
            return False

    def big_neighborhood_filter(self, neighbors, layer):
        """Filter neighborhoods with max number of neighbors.

        Limit is set to keep XX% of the neighborhoods untouched. Limit is
        computed at initialization
        """
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def augmentation_transform(self, points, normals=None, verbose=False, is_test=False):
        """Implementation of an augmentation transform for point clouds."""
        R = np.eye(points.shape[1])
        if points.shape[1] == 3:
            if self.cfg.augment_rotation == 'vertical':
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            elif self.cfg.augment_rotation == 'all':
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
                alpha = np.random.rand() * 2 * np.pi
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
        R = R.astype(np.float32)
        min_s = self.cfg.augment_scale_min
        max_s = self.cfg.augment_scale_max
        if self.cfg.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s
        symmetries = np.array(self.cfg.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)
        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.cfg.augment_noise).astype(np.float32)
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise
        if is_test:
            return points, scale, R
        if normals is None:
            return augmented_points, scale, R
        else:
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-06)
            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2] * 0, augmented_points[:, 2] * 0 + 1])]
                show_ModelNet_examples(test_p, test_n, test_l)
            return augmented_points, augmented_normals, scale, R


class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        rotations (list[float]): Rotations of anchors in a feature grid.
    """

    def __init__(self, ranges, sizes=[[1.6, 3.9, 1.56]], rotations=[0, 1.5707963]):
        if len(sizes) != len(ranges):
            assert len(ranges) == 1
            ranges = ranges * len(sizes)
        assert len(ranges) == len(sizes)
        self.sizes = sizes
        self.ranges = ranges
        self.rotations = rotations

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    def grid_anchors(self, featmap_size, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(self.anchors_single_range(featmap_size, anchor_range, anchor_size, self.rotations, device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(self, feature_size, anchor_range, sizes=[[1.6, 3.9, 1.56]], rotations=[0, 1.5707963], device='cuda'):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            torch.Tensor: Anchors with shape                 [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_size[2], device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3)
        rotations = torch.tensor(rotations, device=device)
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        rets = list(rets)
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).unsqueeze(-1)
        sizes = sizes.reshape([1, 1, 1, 1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)
        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        return ret


class BBoxCoder(object):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self):
        super(BBoxCoder, self).__init__()

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl, dr,
        dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def box3d_to_bev(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to BEV in XYWHR format.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        torch.Tensor: Converted BEV boxes in XYWHR format.
    """
    return boxes3d[:, [0, 1, 3, 4, 6]]


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range.             Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of             [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def box3d_to_bev2d(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to neareset BEV without
    rotation.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        torch.Tensor: Converted BEV boxes in XYWH format.
    """
    bev_rotated_boxes = box3d_to_bev(boxes3d)
    rotations = bev_rotated_boxes[:, -1]
    normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))
    conditions = (normed_rotations > np.pi / 4)[..., None]
    bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4])
    centers = bboxes_xywh[:, :2]
    dims = bboxes_xywh[:, 2:]
    bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
    return bev_boxes


def xywhr_to_xyxyr(boxes_xywhr):
    """Convert rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2
    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def multiclass_nms(boxes, scores, score_thr):
    """Multi-class nms for 3D boxes.

    Args:
        boxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        scores (torch.Tensor): Multi-level boxes with shape
            (N, ). N is the number of boxes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.

    Returns:
        list[torch.Tensor]: Return a list of indices after nms,
            with an entry for each class.
    """
    idxs = []
    for i in range(scores.shape[1]):
        cls_inds = scores[:, i] > score_thr
        if not cls_inds.any():
            idxs.append(torch.tensor([], dtype=torch.long, device=cls_inds.device))
            continue
        orig_idx = torch.arange(cls_inds.shape[0], device=cls_inds.device, dtype=torch.long)[cls_inds]
        _scores = scores[cls_inds, i]
        _boxes = boxes[cls_inds, :]
        _bev = xywhr_to_xyxyr(box3d_to_bev(_boxes))
        idx = nms(_bev, _scores, 0.01)
        idxs.append(orig_idx[idx])
    return idxs


class Anchor3DHead(nn.Module):

    def __init__(self, num_classes=1, in_channels=384, feat_channels=384, nms_pre=100, score_thr=0.1, dir_offset=0, ranges=[[0, -40.0, -3, 70.0, 40.0, 1]], sizes=[[0.6, 1.0, 1.5]], rotations=[0, 1.57], iou_thr=[[0.35, 0.5]]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.dir_offset = dir_offset
        self.iou_thr = iou_thr
        if len(self.iou_thr) != num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * num_classes
        assert len(self.iou_thr) == num_classes
        self.anchor_generator = Anchor3DRangeGenerator(ranges=ranges, sizes=sizes, rotations=rotations)
        self.num_anchors = self.anchor_generator.num_base_anchors
        self.bbox_coder = BBoxCoder()
        self.box_code_size = 7
        self.fp16_enabled = False
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)
        self.init_weights()

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """Initialize conv/fc bias value according to giving probablity."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        return bias_init

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = self.bias_init_with_prob(0.01)
        self.normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        self.normal_init(self.conv_reg, std=0.01)

    def forward(self, x):
        """Forward function on a feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox                 regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def assign_bboxes(self, pred_bboxes, target_bboxes):
        """Assigns target bboxes to given anchors.

        Args:
            pred_bboxes (torch.Tensor): Bbox predictions (anchors).
            target_bboxes (torch.Tensor): Bbox targets.

        Returns:
            torch.Tensor: Assigned target bboxes for each given anchor.
            torch.Tensor: Flat index of matched targets.
            torch.Tensor: Index of positive matches.
            torch.Tensor: Index of negative matches.
        """
        anchors = [self.anchor_generator.grid_anchors(pred_bboxes.shape[-2:], device=pred_bboxes.device) for _ in range(len(target_bboxes))]
        anchors_cnt = torch.tensor(anchors[0].shape[:-1]).prod()
        rot_angles = anchors[0].shape[-2]
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):
            """Inject class dimension in the given indices (...

            z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)
            """
            z = idx // rot_angles
            x = idx % rot_angles
            return z * self.num_classes * rot_angles + j * rot_angles + x
        idx_off = 0
        for i in range(len(target_bboxes)):
            for j, (neg_th, pos_th) in enumerate(self.iou_thr):
                anchors_stride = anchors[i][..., j, :, :].reshape(-1, self.box_code_size)
                if target_bboxes[i].shape[0] == 0:
                    assigned_bboxes.append(torch.zeros((0, 7), device=pred_bboxes.device))
                    target_idxs.append(torch.zeros((0,), dtype=torch.long, device=pred_bboxes.device))
                    pos_idxs.append(torch.zeros((0,), dtype=torch.long, device=pred_bboxes.device))
                    neg_idxs.append(torch.zeros((0,), dtype=torch.long, device=pred_bboxes.device))
                    continue
                overlaps = bbox_overlaps(box3d_to_bev2d(target_bboxes[i]), box3d_to_bev2d(anchors_stride))
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                gt_max_overlaps, _ = overlaps.max(dim=1)
                pos_idx = max_overlaps >= pos_th
                neg_idx = (max_overlaps >= 0) & (max_overlaps < neg_th)
                for k in range(len(target_bboxes[i])):
                    if gt_max_overlaps[k] >= neg_th:
                        pos_idx[overlaps[k, :] == gt_max_overlaps[k]] = True
                assigned_bboxes.append(self.bbox_coder.encode(anchors_stride[pos_idx], target_bboxes[i][argmax_overlaps[pos_idx]]))
                target_idxs.append(argmax_overlaps[pos_idx] + idx_off)
                pos_idx = flatten_idx(pos_idx.nonzero(as_tuple=False).squeeze(-1), j) + i * anchors_cnt
                neg_idx = flatten_idx(neg_idx.nonzero(as_tuple=False).squeeze(-1), j) + i * anchors_cnt
                pos_idxs.append(pos_idx)
                neg_idxs.append(neg_idx)
            idx_off += len(target_bboxes[i])
        return torch.cat(assigned_bboxes, axis=0), torch.cat(target_idxs, axis=0), torch.cat(pos_idxs, axis=0), torch.cat(neg_idxs, axis=0)

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds, dir_preds):
            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)
            bboxes.append(b)
            scores.append(s)
            labels.append(l)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        assert cls_scores.size()[-2:] == dir_preds.size()[-2:]
        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:], device=cls_scores.device)
        anchors = anchors.reshape(-1, self.box_code_size)
        dir_preds = dir_preds.permute(1, 2, 0).reshape(-1, 2)
        dir_scores = torch.max(dir_preds, dim=-1)[1]
        cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_scores.sigmoid()
        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, self.box_code_size)
        if scores.shape[0] > self.nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(self.nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_scores = dir_scores[topk_inds]
        bboxes = self.bbox_coder.decode(anchors, bbox_preds)
        idxs = multiclass_nms(bboxes, scores, self.score_thr)
        labels = [torch.full((len(idxs[i]),), i, dtype=torch.long) for i in range(self.num_classes)]
        labels = torch.cat(labels)
        scores = [scores[idxs[i], i] for i in range(self.num_classes)]
        scores = torch.cat(scores)
        idxs = torch.cat(idxs)
        bboxes = bboxes[idxs]
        dir_scores = dir_scores[idxs]
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, 1, np.pi)
            bboxes[..., 6] = dir_rot + self.dir_offset + np.pi * dir_scores
        return bboxes, scores, labels


class BoundingBox3D:
    """Class that defines an axially-oriented bounding box."""
    next_id = 1

    def __init__(self, center, front, up, left, size, label_class, confidence, meta=None, show_class=False, show_confidence=False, show_meta=None, identifier=None, arrow_length=1.0):
        """Creates a bounding box.

        Front, up, left define the axis of the box and must be normalized and
        mutually orthogonal.

        Args:
            center: (x, y, z) that defines the center of the box.
            front: normalized (i, j, k) that defines the front direction of the
                box.
            up: normalized (i, j, k) that defines the up direction of the box.
            left: normalized (i, j, k) that defines the left direction of the
                box.
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge.
            label_class: integer specifying the classification label. If an LUT
                is specified in create_lines() this will be used to determine
                the color of the box.
            confidence: confidence level of the box.
            meta: a user-defined string (optional).
            show_class: displays the class label in text near the box
                (optional).
            show_confidence: displays the confidence value in text near the box
                (optional).
            show_meta: displays the meta string in text near the box (optional).
            identifier: a unique integer that defines the id for the box
                (optional, will be generated if not provided).
            arrow_length: the length of the arrow in the front_direct. Set to
                zero to disable the arrow (optional).
        """
        assert len(center) == 3
        assert len(front) == 3
        assert len(up) == 3
        assert len(left) == 3
        assert len(size) == 3
        self.center = np.array(center, dtype='float32')
        self.front = np.array(front, dtype='float32')
        self.up = np.array(up, dtype='float32')
        self.left = np.array(left, dtype='float32')
        self.size = size
        self.label_class = label_class
        self.confidence = confidence
        self.meta = meta
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_meta = show_meta
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = 'box:' + str(BoundingBox3D.next_id)
            BoundingBox3D.next_id += 1
        self.arrow_length = arrow_length

    def __repr__(self):
        s = str(self.identifier) + ' (class=' + str(self.label_class) + ', conf=' + str(self.confidence)
        if self.meta is not None:
            s = s + ', meta=' + str(self.meta)
        s = s + ')'
        return s

    @staticmethod
    def create_lines(boxes, lut=None, out_format='lineset'):
        """Creates a LineSet that can be used to render the boxes.

        Args:
            boxes: the list of bounding boxes
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
            out_format (str): Output format. Can be "lineset" (default) for the
                Open3D lineset or "dict" for a dictionary of lineset properties.

        Returns:
            For out_format == "lineset": open3d.geometry.LineSet
            For out_format == "dict": Dictionary of lineset properties
                ("vertex_positions", "line_indices", "line_colors", "bbox_labels",
                "bbox_confidences").
        """
        if out_format not in ('lineset', 'dict'):
            raise ValueError("Please specify an output_format of 'lineset' (default) or 'dict'.")
        nverts = 14
        nlines = 17
        points = np.zeros((nverts * len(boxes), 3), dtype='float32')
        indices = np.zeros((nlines * len(boxes), 2), dtype='int32')
        colors = np.zeros((nlines * len(boxes), 3), dtype='float32')
        for i, box in enumerate(boxes):
            pidx = nverts * i
            x = 0.5 * box.size[0] * box.left
            y = 0.5 * box.size[1] * box.up
            z = 0.5 * box.size[2] * box.front
            arrow_tip = box.center + z + box.arrow_length * box.front
            arrow_mid = box.center + z + 0.6 * box.arrow_length * box.front
            head_length = 0.3 * box.arrow_length
            points[pidx] = box.center + x + y + z
            points[pidx + 1] = box.center - x + y + z
            points[pidx + 2] = box.center - x + y - z
            points[pidx + 3] = box.center + x + y - z
            points[pidx + 4] = box.center + x - y + z
            points[pidx + 5] = box.center - x - y + z
            points[pidx + 6] = box.center - x - y - z
            points[pidx + 7] = box.center + x - y - z
            points[pidx + 8] = box.center + z
            points[pidx + 9] = arrow_tip
            points[pidx + 10] = arrow_mid + head_length * box.up
            points[pidx + 11] = arrow_mid - head_length * box.up
            points[pidx + 12] = arrow_mid + head_length * box.left
            points[pidx + 13] = arrow_mid - head_length * box.left
        for i, box in enumerate(boxes):
            pidx = nverts * i
            idx = nlines * i
            indices[idx:idx + nlines] = (pidx, pidx + 1), (pidx + 1, pidx + 2), (pidx + 2, pidx + 3), (pidx + 3, pidx), (pidx + 4, pidx + 5), (pidx + 5, pidx + 6), (pidx + 6, pidx + 7), (pidx + 7, pidx + 4), (pidx + 0, pidx + 4), (pidx + 1, pidx + 5), (pidx + 2, pidx + 6), (pidx + 3, pidx + 7), (pidx + 8, pidx + 9), (pidx + 9, pidx + 10), (pidx + 9, pidx + 11), (pidx + 9, pidx + 12), (pidx + 9, pidx + 13)
            if lut is not None and box.label_class in lut.labels:
                label = lut.labels[box.label_class]
                c = label.color[0], label.color[1], label.color[2]
            elif box.confidence == -1.0:
                c = 0.0, 1.0, 0.0
            elif box.confidence >= 0 and box.confidence <= 1.0:
                c = 1.0, 0.0, 0.0
            else:
                c = 0.5, 0.5, 0.5
            colors[idx:idx + nlines] = c
        if out_format == 'lineset':
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector(indices)
            lines.colors = o3d.utility.Vector3dVector(colors)
        elif out_format == 'dict':
            lines = {'vertex_positions': points, 'line_indices': indices, 'line_colors': colors, 'bbox_labels': tuple(b.label_class for b in boxes), 'bbox_confidences': tuple(b.confidence for b in boxes)}
        return lines

    @staticmethod
    def project_to_img(boxes, img, lidar2img_rt=np.ones(4), lut=None):
        """Returns image with projected 3D bboxes

        Args:
            boxes: the list of bounding boxes
            img: an RGB image
            lidar2img_rt: 4x4 transformation from lidar frame to image plane
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
        """
        lines = BoundingBox3D.create_lines(boxes, lut, out_format='dict')
        points = lines['vertex_positions']
        indices = lines['line_indices']
        colors = lines['line_colors']
        pts_4d = np.concatenate([points.reshape(-1, 3), np.ones((len(boxes) * 14, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-05, a_max=100000.0)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(len(boxes), 14, 2)
        indices_2d = indices[..., :2].reshape(len(boxes), 17, 2)
        colors_2d = colors[..., :3].reshape(len(boxes), 17, 3)
        return BoundingBox3D.plot_rect3d_on_img(img, len(boxes), imgfov_pts_2d, indices_2d, colors_2d, thickness=3)

    @staticmethod
    def plot_rect3d_on_img(img, num_rects, rect_corners, line_indices, color=None, thickness=1):
        """Plot the boundary lines of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2] or
                [num_rect, 14, 2] if counting arrows.
            line_indices (numpy.array): indicates connectivity of lines between
                rect_corners.  Should be in the shape of [num_rect, 12, 2] or
                [num_rect, 17, 2] if counting arrows.
            color (tuple[int]): The color to draw bboxes. Default: (1.0, 1.0,
                1.0), i.e. white.
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        if color is None:
            color = np.ones((line_indices.shape[0], line_indices.shape[1], 3))
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int)
            interesting_corners_scale = 3.0
            if min(corners[:, 0]) < -interesting_corners_scale * img.shape[1] or max(corners[:, 0]) > interesting_corners_scale * img.shape[1] or min(corners[:, 1]) < -interesting_corners_scale * img.shape[0] or max(corners[:, 1]) > interesting_corners_scale * img.shape[0]:
                continue
            for j, (start, end) in enumerate(line_indices[i]):
                c = tuple(color[i][j] * 255)
                c = int(c[0]), int(c[1]), int(c[2])
                if i != 0:
                    pt1 = corners[start % (14 * i), 0], corners[start % (14 * i), 1]
                    pt2 = corners[end % (14 * i), 0], corners[end % (14 * i), 1]
                else:
                    pt1 = corners[start, 0], corners[start, 1]
                    pt2 = corners[end, 0], corners[end, 1]
                draw.line([pt1, pt2], fill=c, width=thickness)
        return np.array(img_pil).astype(np.uint8)


class BEVBox3D(BoundingBox3D):
    """Class that defines a special bounding box for object detection, with only
    one rotation axis (yaw).

                            up z    x front (yaw=0.5*pi)
                                ^   ^
                                |  /
                                | /
        (yaw=pi) left y <------ 0

    The relative coordinate of bottom center in a BEV box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and increases from
    the negative direction of y to the positive direction of x.
    """

    def __init__(self, center, size, yaw, label_class, confidence, world_cam=None, cam_img=None, **kwargs):
        """Creates a bounding box.

        Args:
            center: (x, y, z) that defines the center of the box
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge
            yaw: yaw angle of box
            label_class: integer specifying the classification label. If an LUT is
                specified in create_lines() this will be used to determine the color
                of the box.
            confidence: confidence level of the box
            world_cam: world to camera transformation
            cam_img: camera to image transformation
        """
        self.yaw = yaw
        self.world_cam = world_cam
        self.cam_img = cam_img
        left = [np.cos(self.yaw), -np.sin(self.yaw), 0]
        front = [np.sin(self.yaw), np.cos(self.yaw), 0]
        up = [0, 0, 1]
        super().__init__(center, front, up, left, size, label_class, confidence, **kwargs)
        self.points_inside_box = np.array([])
        self.level = self.get_difficulty()
        self.dis_to_cam = np.linalg.norm(self.to_camera()[:3])

    def to_kitti_format(self, score=1.0):
        """This method transforms the class to KITTI format."""
        box2d = self.to_img()
        box2d[2:] += box2d[:2]
        truncation = -1
        occlusion = -1
        box = self.to_camera()
        center = box[:3]
        size = box[3:6]
        ry = box[6]
        x, z = center[0], center[2]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (self.label_class, truncation, occlusion, alpha, box2d[0], box2d[1], box2d[2], box2d[3], size[0], size[1], size[2], center[0], center[1], center[2], ry, score)
        return kitti_str

    def generate_corners3d(self):
        """Generate corners3d representation for this object.

        Returns:
            corners_3d: (8, 3) corners of box3d in camera coordinates.
        """
        w, h, l = self.size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        R = np.array([[np.cos(self.yaw), 0, np.sin(self.yaw)], [0, 1, 0], [-np.sin(self.yaw), 0, np.cos(self.yaw)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.to_camera()[:3]
        return corners3d

    def to_xyzwhlr(self):
        """Returns box in the common 7-sized vector representation: (x, y, z, w,
        l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
        the width, length and height of the box a is the yaw angle.

        Returns:
            box(7,)

        """
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[3:6] = np.array(self.size)[[0, 2, 1]]
        bbox[6] = self.yaw
        return bbox

    def to_camera(self):
        """Transforms box into camera space.

                     up x    y front
                        ^   ^
                        |  /
                        | /
         left z <------ 0

        Returns box in the common 7-sized vector representation.
        (x, y, z, l, h, w, a), where
        (x, y, z) is the bottom center of the box,
        (l, h, w) is the length, height, width of the box
        a is the yaw angle

        Returns:
            transformed box: (7,)
        """
        if self.world_cam is None:
            return self.to_xyzwhlr()[[1, 2, 0, 4, 5, 3, 6]]
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[0:3] = (np.array([*bbox[0:3], 1.0]) @ self.world_cam)[:3]
        bbox[3:6] = [self.size[1], self.size[0], self.size[2]]
        bbox[6] = self.yaw
        return bbox

    def to_img(self):
        """Transforms box into 2d box.

        Returns:
            transformed box: (4,)
        """
        if self.cam_img is None:
            return None
        corners = self.generate_corners3d()
        corners = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=-1)
        bbox_img = np.matmul(corners, self.cam_img)
        bbox_img = bbox_img[:, :2] / bbox_img[:, 3:]
        minxy = np.min(bbox_img, axis=0)
        maxxy = np.max(bbox_img, axis=0)
        size = maxxy - minxy
        center = minxy + size / 2
        return np.concatenate([center, size])

    def get_difficulty(self):
        """General method to compute difficulty, can be overloaded.

        Returns:
            Difficulty depending on projected height of box.
        """
        if self.cam_img is None:
            return 0
        heights = [40, 25]
        height = self.to_img()[3] + 1
        diff = -1
        for j in range(len(heights)):
            if height >= heights[j]:
                diff = j
                break
        return diff

    def to_dict(self):
        """Convert data for evaluation:"""
        return {'bbox': self.to_camera(), 'label': self.label_class, 'score': self.confidence, 'difficulty': self.level}

    @staticmethod
    def to_dicts(bboxes):
        """Convert data for evaluation:

        Args:
            bboxes: List of BEVBox3D bboxes.
        """
        box_dicts = {'bbox': np.empty((len(bboxes), 7)), 'label': np.empty((len(bboxes),), dtype='<U20'), 'score': np.empty((len(bboxes),)), 'difficulty': np.empty((len(bboxes),))}
        for i in range(len(bboxes)):
            box_dict = bboxes[i].to_dict()
            for k in box_dict:
                box_dicts[k][i] = box_dict[k]
        return box_dicts


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss."""

    def __init__(self, loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.cross_entropy(cls_score, label, reduction='none')
        if weight is not None:
            loss = loss * weight
        loss = loss * self.loss_weight
        if avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.mean()


def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index == out_idx).float()


class FocalLoss(nn.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        pred_sigmoid = pred.sigmoid()
        if len(pred.shape) > 1:
            target = one_hot(target, int(pred.shape[-1]))
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        if weight is not None:
            loss = loss * weight
        loss = loss * self.loss_weight
        if avg_factor is None:
            return loss.mean()
        elif avg_factor > 0:
            return loss.sum() / avg_factor
        else:
            return loss


class Augmentation:
    """Class consisting common augmentation methods for different pipelines."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def recenter(self, data, cfg):
        """Recenter pointcloud/features to origin.

        Typically used before rotating the pointcloud.

        Args:
            data: Pointcloud or features.
            cfg: config dict where
                Key 'dim' specifies dimension to be recentered.

        """
        if not cfg:
            return data
        dim = cfg.get('dim', [0, 1, 2])
        data[:, dim] = data[:, dim] - data.mean(0)[dim]
        return data

    def normalize(self, pc, feat, cfg):
        """Normalize pointcloud and/or features.

        Points are normalized in [0, 1] and features can take custom
        scale and bias.

        Args:
            pc: Pointcloud.
            feat: features.
            cfg: configuration dictionary.

        """
        if 'points' in cfg:
            cfg_p = cfg['points']
            if cfg_p.get('method', 'linear') == 'linear':
                pc -= pc.mean(0)
                pc /= (pc.max(0) - pc.min(0)).max()
            else:
                raise ValueError(f"Unsupported method : {cfg_p.get('method')}")
        if 'feat' in cfg and feat is not None:
            cfg_f = cfg['feat']
            if cfg_f.get('method', 'linear') == 'linear':
                bias = cfg_f.get('bias', 0)
                scale = cfg_f.get('scale', 1)
                feat -= bias
                feat /= scale
            else:
                raise ValueError(f"Unsupported method : {cfg_f.get('method')}")
        return pc, feat

    def rotate(self, pc, cfg):
        """Rotate the pointcloud.

        Two methods are supported. `vertical` rotates the pointcloud
        along yaw. `all` randomly rotates the pointcloud in all directions.

        Args:
            pc: Pointcloud to augment.
            cfg: configuration dictionary.

        """
        if np.abs(pc[:, :2].mean()) > 0.01:
            warnings.warn(f'It is recommended to recenter the pointcloud before calling rotate.')
        method = cfg.get('method', 'vertical')
        if method == 'vertical':
            theta = self.rng.random() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        elif method == 'all':
            theta = self.rng.random() * 2 * np.pi
            phi = (self.rng.random() - 0.5) * np.pi
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            alpha = self.rng.random() * 2 * np.pi
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
        else:
            raise ValueError(f'Unsupported method : {method}')
        R = R.astype(np.float32)
        return np.matmul(pc, R)

    def scale(self, pc, cfg):
        """Scale augmentation for pointcloud.

        If `scale_anisotropic` is True, each point is scaled differently.
        else, same scale from range ['min_s', 'max_s') is applied to each point.

        Args:
            pc: Pointcloud to scale.
            cfg: configuration dict.

        """
        scale_anisotropic = cfg.get('scale_anisotropic', False)
        min_s = cfg.get('min_s', 1.0)
        max_s = cfg.get('max_s', 1.0)
        if scale_anisotropic:
            scale = self.rng.random(pc.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = self.rng.random() * (max_s - min_s) + min_s
        return pc * scale

    def noise(self, pc, cfg):
        noise_std = cfg.get('noise_std', 0.001)
        noise = (self.rng.standard_normal((pc.shape[0], pc.shape[1])) * noise_std).astype(np.float32)
        return pc + noise

    def augment(self, data):
        raise NotImplementedError('Please use one of SemsegAugmentation or ObjdetAugmentation.')


def filter_by_min_points(bboxes, min_points_dict):
    """Filter ground truths by number of points in the bbox."""
    filtered_boxes = []
    for box in bboxes:
        if box.label_class in min_points_dict.keys():
            if box.points_inside_box.shape[0] > min_points_dict[box.label_class]:
                filtered_boxes.append(box)
        else:
            filtered_boxes.append(box)
    return filtered_boxes


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=2):
    """Rotate points in specific axis.

    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.

    Returns:
        np.ndarray: Rotated points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros], [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros], [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin], [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 1.0, 0.5)):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corner_to_surfaces_3d(corners):
    """Convert 3d box corners from corner function above to surfaces that normal
    vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    surfaces = np.array([[corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]], [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]], [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]], [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]], [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]], [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]]]).transpose([2, 0, 1, 3])
    return surfaces


def surface_equ_3d(polygon_surfaces):
    """Compute normal vectors for polygon surfaces.

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def points_in_convex_polygon_3d(points, polygon_surfaces, num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of             (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).             All surfaces' normal vector must direct to internal.             Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains             shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    points = np.reshape(points, (num_points, 1, 1, 3))
    normal_vec = np.reshape(normal_vec, (1, num_polygons, max_num_surfaces, 3))
    num_surfaces = np.reshape(num_surfaces, (num_polygons, 1))
    sign = np.sum(points * normal_vec, axis=-1) + d
    out_range = np.arange(max_num_surfaces) >= num_surfaces
    out_range = np.reshape(out_range, (1, num_polygons, max_num_surfaces))
    ret = np.all(sign < 0 | out_range, axis=-1)
    return ret


def points_in_box(points, rbbox, origin=(0.5, 0.5, 0), camera_frame=False, cam_world=None):
    """Check points in rotated bbox and return indices.

    If `rbbox` is in camera frame, it is first converted to world frame using
    `cam_world`. Returns a 2D array classifying each point for each box.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation (camera/world frame).
        origin (tuple[int]): Indicate the position of box center.
        camera_frame: True if `rbbox` are in camera frame(like kitti format, where y
          coordinate is height), False for [x, y, z, dx, dy, dz, yaw] format.
        cam_world: camera to world transformation matrix. Required when `camera_frame` is True.

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    if len(rbbox) == 0:
        return np.zeros((0, 7))
    if camera_frame:
        assert cam_world is not None, 'Provide cam_to_world matrix if points are in camera frame.'
        points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
        points = np.matmul(points, cam_world)[..., :3]
    rbbox = np.array(rbbox)
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d(points[:, :3], surfaces)
    return indices


def remove_points_in_boxes(points, boxes):
    """Remove the points in the sampled bounding boxes.

    Args:
        points (np.ndarray): Input point cloud array.
        boxes (np.ndarray): Sampled ground truth boxes.

    Returns:
        np.ndarray: Points with those in the boxes removed.
    """
    flat_boxes = [box.to_xyzwhlr() for box in boxes]
    masks = points_in_box(points, flat_boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def box_collision_test(boxes, qboxes):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
    """
    boxes = np.array([box.to_xyzwhlr() for box in boxes], dtype=np.float32)
    qboxes = np.array([box.to_xyzwhlr() for box in qboxes], dtype=np.float32)
    boxes = boxes[:, [0, 1, 3, 4, 6]]
    qboxes = qboxes[:, [0, 1, 3, 4, 6]]
    coll_mat = iou_bev(boxes, qboxes)
    coll_mat[coll_mat != 0] = 1
    return coll_mat.astype(np.bool)


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape             (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box2d(boxes, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4, 2))
    flat_boxes = np.array([box.to_xyzwhlr() for box in boxes])
    centers = flat_boxes[:, 0:2]
    dims = flat_boxes[:, 3:5]
    angles = flat_boxes[:, 6]
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def random_sample(files, num):
    if len(files) <= num:
        return files
    return random.sample(files, num)


def sample_class(class_name, num, gt_boxes, db_boxes):
    if num == 0:
        return []
    sampled = random_sample(db_boxes, num)
    sampled = copy.deepcopy(sampled)
    num_gt = len(gt_boxes)
    num_sampled = len(sampled)
    gt_boxes_bev = center_to_corner_box2d(gt_boxes)
    boxes = (gt_boxes + sampled).copy()
    coll_mat = box_collision_test(boxes, boxes)
    diag = np.arange(len(boxes))
    coll_mat[diag, diag] = False
    valid_samples = []
    for i in range(num_gt, num_gt + num_sampled):
        if coll_mat[i].any():
            coll_mat[i] = False
            coll_mat[:, i] = False
        else:
            valid_samples.append(sampled[i - num_gt])
    return valid_samples


class ObjdetAugmentation(Augmentation):
    """Class consisting different augmentation for Object Detection"""

    def __init__(self, cfg, seed=None):
        super(ObjdetAugmentation, self).__init__(cfg, seed=seed)
        all_methods = ['recenter', 'normalize', 'rotate', 'scale', 'noise', 'PointShuffle', 'ObjectRangeFilter', 'ObjectSample']
        for method in cfg:
            if method not in all_methods:
                warnings.warn(f'Augmentation method : {method} does not exist. Please verify!')

    def PointShuffle(self, data):
        """Shuffle Pointcloud."""
        self.rng.shuffle(data['point'])
        return data

    @staticmethod
    def in_range_bev(box_range, box):
        return (box[0] > box_range[0]) & (box[1] > box_range[1]) & (box[0] < box_range[2]) & (box[1] < box_range[3])

    def ObjectRangeFilter(self, data, pcd_range):
        """Filter Objects in the given range."""
        pcd_range = np.array(pcd_range)
        bev_range = pcd_range[[0, 1, 3, 4]]
        filtered_boxes = []
        for box in data['bounding_boxes']:
            if self.in_range_bev(bev_range, box.to_xyzwhlr()):
                filtered_boxes.append(box)
        return {'point': data['point'], 'bounding_boxes': filtered_boxes, 'calib': data['calib']}

    def ObjectSample(self, data, db_boxes_dict, sample_dict):
        """Increase frequency of objects in a pointcloud.

        Randomly place objects in a pointcloud from a database of
        all objects within the dataset. Checks collision with existing objects.

        Args:
            data: Input data dict with keys ('point', 'bounding_boxes', 'calib').
            db_boxes_dict: dict for different objects.
            sample_dict: dict for number of objects to sample.

        """
        rate = 1.0
        points = data['point']
        bboxes = data['bounding_boxes']
        gt_labels_3d = [box.label_class for box in data['bounding_boxes']]
        sampled_num_dict = {}
        for class_name in sample_dict.keys():
            max_sample_num = sample_dict[class_name]
            existing = np.sum([(n == class_name) for n in gt_labels_3d])
            sampled_num = int(max_sample_num - existing)
            sampled_num = np.round(rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
        sampled = []
        for class_name in sampled_num_dict.keys():
            sampled_num = sampled_num_dict[class_name]
            if sampled_num < 0:
                continue
            sampled_cls = sample_class(class_name, sampled_num, bboxes, db_boxes_dict[class_name])
            sampled += sampled_cls
            bboxes = bboxes + sampled_cls
        if len(sampled) != 0:
            sampled_points = np.concatenate([box.points_inside_box for box in sampled], axis=0)
            points = remove_points_in_boxes(points, sampled)
            points = np.concatenate([sampled_points[:, :4], points], axis=0)
        return {'point': points, 'bounding_boxes': bboxes, 'calib': data['calib']}

    def load_gt_database(self, pickle_path, min_points_dict, sample_dict):
        """Load ground truth object database.

        Args:
            pickle_path: Path of pickle file generated using `scripts/collect_bbox.py`.
            min_points_dict: A dictionary to filter objects based on number of points inside.
                Format of dict {'class_name': num_points}.
            sample_dict: A dictionary to decide number of objects to sample.
                Format of dict {'class_name': num_instance}

        """
        db_boxes = pickle.load(open(pickle_path, 'rb'))
        if min_points_dict is not None:
            db_boxes = filter_by_min_points(db_boxes, min_points_dict)
        db_boxes_dict = {}
        for key in sample_dict.keys():
            db_boxes_dict[key] = []
        for db_box in db_boxes:
            if db_box.label_class in sample_dict.keys():
                db_boxes_dict[db_box.label_class].append(db_box)
        self.db_boxes_dict = db_boxes_dict

    def augment(self, data, attr, seed=None):
        """Augment object detection data.

        Available augmentations are:
            `ObjectSample`: Insert objects from ground truth database.
            `ObjectRangeFilter`: Filter pointcloud from given bounds.
            `PointShuffle`: Shuffle the pointcloud.

        Args:
            data: A dictionary object returned from the dataset class.
            attr: Attributes for current pointcloud.

        Returns:
            Augmented `data` dictionary.

        """
        cfg = self.cfg
        if cfg is None:
            return data
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if 'recenter' in cfg:
            if cfg['recenter']:
                data['point'] = self.recenter(data['point'], cfg['recenter'])
        if 'normalize' in cfg:
            data['point'], _ = self.normalize(data['point'], None, cfg['normalize'])
        if 'rotate' in cfg:
            data['point'] = self.rotate(data['point'], cfg['rotate'])
        if 'scale' in cfg:
            data['point'] = self.scale(data['point'], cfg['scale'])
        if 'noise' in cfg:
            data['point'] = self.noise(data['point'], cfg['noise'])
        if 'ObjectSample' in cfg:
            if not hasattr(self, 'db_boxes_dict'):
                data_path = attr['path']
                for _ in range(3):
                    data_path = os.path.split(data_path)[0]
                pickle_path = os.path.join(data_path, 'bboxes.pkl')
                if 'pickle_path' not in cfg['ObjectSample']:
                    cfg['ObjectSample']['pickle_path'] = pickle_path
                self.load_gt_database(**cfg['ObjectSample'])
            data = self.ObjectSample(data, db_boxes_dict=self.db_boxes_dict, sample_dict=cfg['ObjectSample']['sample_dict'])
        if cfg.get('ObjectRangeFilter', False):
            data = self.ObjectRangeFilter(data, cfg['ObjectRangeFilter']['point_cloud_range'])
        if cfg.get('PointShuffle', False):
            data = self.PointShuffle(data)
        return data


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self, in_channels, out_channels, last_layer=False, mode='max'):
        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.norm = nn.BatchNorm1d(self.units, eps=0.001, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.0, 40, 1).
    """

    def __init__(self, in_channels=4, feat_channels=(64,), voxel_size=(0.16, 0.16, 4), point_cloud_range=(0, -40.0, -3, 70.0, 40.0, 1)):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        in_channels += 5
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, last_layer=last_layer, mode='max'))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.fp16_enabled = False
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)
        f_center = features[:, :, :2].clone().detach()
        f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
        features_ls.append(f_center)
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)
        return features.squeeze(dim=1)


class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels=64, output_shape=[496, 432]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    def forward(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.in_channels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
        return batch_canvas


class PointPillarsVoxelization(torch.nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points=32, max_voxels=[16000, 40000]):
        """Voxelization layer for the PointPillars model.

        Args:
            voxel_size: voxel edge lengths with format [x, y, z].
            point_cloud_range: The valid range of point coordinates as
                [x_min, y_min, z_min, x_max, y_max, z_max].
            max_num_points: The maximum number of points per voxel.
            max_voxels: The maximum number of voxels. May be a tuple with
                values for training and testing.
        """
        super().__init__()
        self.voxel_size = torch.Tensor(voxel_size)
        self.point_cloud_range = point_cloud_range
        self.points_range_min = torch.Tensor(point_cloud_range[:3])
        self.points_range_max = torch.Tensor(point_cloud_range[3:])
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple) or isinstance(max_voxels, list):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)

    def forward(self, points_feats):
        """Forward function.

        Args:
            points_feats: Tensor with point coordinates and features. The shape
                is [N, 3+C] with N as the number of points and C as the number
                of feature channels.

        Returns:
            (out_voxels, out_coords, out_num_points).
            * out_voxels is a dense list of point coordinates and features for
              each voxel. The shape is [num_voxels, max_num_points, 3+C].
            * out_coords is tensor with the integer voxel coords and shape
              [num_voxels,3]. Note that the order of dims is [z,y,x].
            * out_num_points is a 1D tensor with the number of points for each
              voxel.
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        points = points_feats[:, :3]
        num_voxels = ((self.points_range_max - self.points_range_min) / self.voxel_size).type(torch.int32)
        ans = voxelize(points, torch.LongTensor([0, points.shape[0]]), self.voxel_size, self.points_range_min, self.points_range_max, self.max_num_points, max_voxels)
        feats = torch.cat([torch.zeros_like(points_feats[0:1, :]), points_feats])
        voxels_point_indices_dense = ragged_to_dense(ans.voxel_point_indices, ans.voxel_point_row_splits, self.max_num_points, torch.tensor(-1)) + 1
        out_voxels = feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]
        in_bounds_y = out_coords[:, 1] < num_voxels[1]
        in_bounds_x = out_coords[:, 2] < num_voxels[0]
        in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)
        out_coords = out_coords[in_bounds]
        out_voxels = out_voxels[in_bounds]
        out_num_points = out_num_points[in_bounds]
        return out_voxels, out_coords, out_num_points


class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self, in_channels=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5], layer_strides=[2, 2, 2]):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [nn.Conv2d(in_filters[i], out_channels[i], 3, bias=False, stride=layer_strides[i], padding=1), nn.BatchNorm2d(out_channels[i], eps=0.001, momentum=0.01), nn.ReLU(inplace=True)]
            for j in range(layer_num):
                block.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                block.append(nn.BatchNorm2d(out_channels[i], eps=0.001, momentum=0.01))
                block.append(nn.ReLU(inplace=True))
            block = nn.Sequential(*block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self, in_channels=[64, 128, 256], out_channels=[128, 128, 128], upsample_strides=[1, 2, 4], use_conv_for_no_stride=False):
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or stride == 1 and not use_conv_for_no_stride:
                upsample_layer = nn.ConvTranspose2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=upsample_strides[i], stride=upsample_strides[i], bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=stride, stride=stride, bias=False)
            deblock = nn.Sequential(upsample_layer, nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.01), nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        if weight is not None:
            loss = loss * weight
        loss = loss * self.loss_weight
        if avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.mean()


class PointPillars(BaseModel):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "PointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self, name='PointPillars', device='cuda', point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1], classes=['car'], voxelize={}, voxel_encoder={}, scatter={}, backbone={}, neck={}, head={}, loss={}, **kwargs):
        super().__init__(name=name, point_cloud_range=point_cloud_range, device=device, **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.lbl2name = {i: n for i, n in enumerate(classes)}
        self.augmenter = ObjdetAugmentation(self.cfg.augment, seed=self.rng)
        self.voxel_layer = PointPillarsVoxelization(point_cloud_range=point_cloud_range, **voxelize)
        self.voxel_encoder = PillarFeatureNet(point_cloud_range=point_cloud_range, **voxel_encoder)
        self.middle_encoder = PointPillarsScatter(**scatter)
        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(num_classes=len(self.classes), **head)
        self.loss_cls = FocalLoss(**loss.get('focal', {}))
        self.loss_bbox = SmoothL1Loss(**loss.get('smooth_l1', {}))
        self.loss_dir = CrossEntropyLoss(**loss.get('cross_entropy', {}))
        self.device = device
        self

    def extract_feats(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward(self, inputs):
        inputs = inputs.point
        x = self.extract_feats(inputs)
        outs = self.bbox_head(x)
        return outs

    def get_optimizer(self, cfg):
        optimizer = torch.optim.AdamW(self.parameters(), **cfg)
        return optimizer, None

    def get_loss(self, results, inputs):
        scores, bboxes, dirs = results
        gt_labels = inputs.labels
        gt_bboxes = inputs.bboxes
        target_bboxes, target_idx, pos_idx, neg_idx = self.bbox_head.assign_bboxes(bboxes, gt_bboxes)
        avg_factor = pos_idx.size(0)
        scores = scores.permute((0, 2, 3, 1)).reshape(-1, self.bbox_head.num_classes)
        target_labels = torch.full((scores.size(0),), self.bbox_head.num_classes, device=scores.device, dtype=gt_labels[0].dtype)
        target_labels[pos_idx] = torch.cat(gt_labels, axis=0)[target_idx]
        loss_cls = self.loss_cls(scores[torch.cat([pos_idx, neg_idx], axis=0)], target_labels[torch.cat([pos_idx, neg_idx], axis=0)], avg_factor=avg_factor)
        cond = (target_labels[pos_idx] >= 0) & (target_labels[pos_idx] < self.bbox_head.num_classes)
        pos_idx = pos_idx[cond]
        target_idx = target_idx[cond]
        target_bboxes = target_bboxes[cond]
        bboxes = bboxes.permute((0, 2, 3, 1)).reshape(-1, self.bbox_head.box_code_size)[pos_idx]
        dirs = dirs.permute((0, 2, 3, 1)).reshape(-1, 2)[pos_idx]
        if len(pos_idx) > 0:
            target_dirs = torch.cat(gt_bboxes, axis=0)[target_idx][:, -1]
            target_dirs = limit_period(target_dirs, 0, 2 * np.pi)
            target_dirs = (target_dirs / np.pi).long() % 2
            loss_dir = self.loss_dir(dirs, target_dirs, avg_factor=avg_factor)
            r0 = torch.sin(bboxes[:, -1:]) * torch.cos(target_bboxes[:, -1:])
            r1 = torch.cos(bboxes[:, -1:]) * torch.sin(target_bboxes[:, -1:])
            bboxes = torch.cat([bboxes[:, :-1], r0], axis=-1)
            target_bboxes = torch.cat([target_bboxes[:, :-1], r1], axis=-1)
            loss_bbox = self.loss_bbox(bboxes, target_bboxes, avg_factor=avg_factor)
        else:
            loss_cls = loss_cls.sum()
            loss_bbox = bboxes.sum()
            loss_dir = dirs.sum()
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_dir': loss_dir}

    def preprocess(self, data, attr):
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng
        points = np.array(data['point'][:, 0:4], dtype=np.float32)
        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])
        points = points[np.where(np.all(np.logical_and(points[:, :3] >= min_val, points[:, :3] < max_val), axis=-1))]
        data['point'] = points
        if attr['split'] not in ['test', 'testing', 'val', 'validation']:
            data = self.augmenter.augment(data, attr, seed=rng)
        new_data = {'point': data['point'], 'calib': data['calib']}
        if attr['split'] not in ['test', 'testing']:
            new_data['bbox_objs'] = data['bounding_boxes']
        if 'full_point' in data:
            points = np.array(data['full_point'][:, 0:4], dtype=np.float32)
            min_val = np.array(self.point_cloud_range[:3])
            max_val = np.array(self.point_cloud_range[3:])
            points = points[np.where(np.all(np.logical_and(points[:, :3] >= min_val, points[:, :3] < max_val), axis=-1))]
            new_data['full_point'] = points
        return new_data

    def transform(self, data, attr):
        t_data = {'point': data['point'], 'calib': data['calib']}
        if attr['split'] not in ['test', 'testing']:
            t_data['bbox_objs'] = data['bbox_objs']
            t_data['labels'] = np.array([self.name2lbl.get(bb.label_class, len(self.classes)) for bb in data['bbox_objs']], dtype=np.int64)
            t_data['bboxes'] = np.array([bb.to_xyzwhlr() for bb in data['bbox_objs']], dtype=np.float32)
        return t_data

    def inference_end(self, results, inputs):
        bboxes_b, scores_b, labels_b = self.bbox_head.get_bboxes(*results)
        inference_result = []
        for _calib, _bboxes, _scores, _labels in zip(inputs.calib, bboxes_b, scores_b, labels_b):
            bboxes = _bboxes.cpu().detach().numpy()
            scores = _scores.cpu().detach().numpy()
            labels = _labels.cpu().detach().numpy()
            inference_result.append([])
            world_cam, cam_img = None, None
            if _calib is not None:
                world_cam = _calib.get('world_cam', None)
                cam_img = _calib.get('cam_img', None)
            for bbox, score, label in zip(bboxes, scores, labels):
                dim = bbox[[3, 5, 4]]
                pos = bbox[:3] + [0, 0, dim[1] / 2]
                yaw = bbox[-1]
                name = self.lbl2name.get(label, 'ignore')
                inference_result[-1].append(BEVBox3D(pos, dim, yaw, name, score, world_cam, cam_img))
        return inference_result


class OneCycleScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scheduler class for cyclic learning rate scheduling.

    Args:
        total_step: number of steps for one cycle.
        lr_max: maximum cyclic learning rate.
        div_factor: factor by which initial learning starts.
    """

    def __init__(self, total_step, lr_max=0.002, div_factor=10.0):
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.total_step = total_step
        super(OneCycleScheduler, self).__init__()

    def __call__(self, step):
        lr_low = self.lr_max / self.div_factor
        angle = np.pi / self.total_step * step
        lr1 = tf.abs(lr_low + (self.lr_max - lr_low) * tf.math.sin(angle))
        angle = np.pi / self.total_step * ((step - self.total_step / 2) % self.total_step)
        lr2 = tf.abs(self.lr_max * tf.math.cos(angle))
        lr = tf.where(step < self.total_step / 2, lr1, lr2)
        return lr


def is_tuple(x) ->bool:
    return isinstance(x, tuple)


def listify(p=None, q=None):
    """Make `p` listy and the same length as `q`."""
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d


def split_bn_bias(layer_groups):
    """Split the layers in `layer_groups` into batchnorm (`bn_types`) and non-
    batchnorm groups.
    """
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, bn_types):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups


def trainable_params(m: nn.Module):
    """Return list of trainable params in `m`."""
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


class OptimWrapper:
    """Basic wrapper around `opt` to simplify hyper-parameters changes."""

    def __init__(self, opt, wd, true_wd: bool=False, bn_wd: bool=True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        """Create an `optim.Optimizer` from `opt_func` with `lr`.

        Set lr on `layer_groups`.
        """
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        return opt

    def new(self, layer_groups):
        """Create a new `OptimWrapper` from `self` with another `layer_groups`
        but the same hyper-parameters.
        """
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        return self.create(opt_func, self.lr, layer_groups, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def __repr__(self) ->str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    def step(self) ->None:
        """Set weight decay and step optimizer."""
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for p in pg1['params']:
                    if p.requires_grad is False:
                        continue
                    p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']:
                        if p.requires_grad is False:
                            continue
                        p.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) ->None:
        """Clear optimizer gradients."""
        self.opt.zero_grad()

    def __getattr__(self, k: str):
        return getattr(self.opt, k, None)

    def clear(self):
        """Reset the state of the inner optimizer."""
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    @property
    def lr(self) ->float:
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) ->None:
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self) ->float:
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) ->None:
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) ->float:
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) ->None:
        """Set beta, or alpha as makes sense, for given optimizer."""
        if val is None:
            return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) ->float:
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) ->None:
        """Set weight decay."""
        if not self.true_wd:
            self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    def read_defaults(self) ->None:
        """Read the values inside the optimizer for the hyper-parameters."""
        self._beta = None
        if 'lr' in self.opt_keys:
            self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys:
            self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys:
            self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys:
            self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys:
            self._wd = self.read_val('weight_decay')

    def set_val(self, key: str, val, bn_groups: bool=True):
        """Set `val` inside the optimizer dictionary at `key`."""
        if is_tuple(val):
            val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups:
                pg2[key] = v
        return val

    def read_val(self, key: str):
        """Read a hyperparameter `key` in the optimizer dictionary."""
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]):
            val = [o[0] for o in val], [o[1] for o in val]
        return val


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, new_xyz=None) ->(torch.Tensor, torch.Tensor):
        """Forward.

        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        if new_xyz is None and self.npoint is not None:
            sampling = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = torch.gather(xyz, 1, torch.stack([sampling] * 3, -1).long())
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping."""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer."""

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False):
        """PointnetSAModule.

        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method, instance_norm=instance_norm)


def rotate_pc_along_y_torch(pc, rot_angle):
    """Rotate point cloud along Y axis.

    Args:
        pc: (N, 3 + C)
        rot_angle: (N)
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)
    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)
    pc_temp = pc[..., [0, 2]].view((pc.shape[0], -1, 2))
    pc[..., [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).view(pc.shape[:-1] + (2,))
    return pc


def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size, get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """Decode bounding box target.

    Args:
        roi_box3d: (N, 7)
        pred_reg: (N, C)
        loc_scope: scope length for x, z loss.
        loc_bin_size: bin size for classifying x, z loss.
        num_head_bin: number of bins for yaw.
        anchor_size: anchor size for proposals.
        get_xz_fine: bool
        get_y_by_bin: bool
        loc_y_scope: float
        loc_y_bin_size: float
        get_ry_fine: bool
    """
    anchor_size = anchor_size
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r
    x_bin = torch.argmax(pred_reg[:, x_bin_l:x_bin_r], dim=1)
    z_bin = torch.argmax(pred_reg[:, z_bin_l:z_bin_r], dim=1)
    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r
        x_res_norm = torch.gather(pred_reg[:, x_res_l:x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l:z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size
        pos_x += x_res
        pos_z += z_res
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r
        y_bin = torch.argmax(pred_reg[:, y_bin_l:y_bin_r], dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l:y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r
        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    ry_bin = torch.argmax(pred_reg[:, ry_bin_l:ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l:ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_ry_fine:
        angle_per_class = np.pi / 2 / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = ry_bin.float() * angle_per_class + angle_per_class / 2 + ry_res - np.pi / 4
    else:
        angle_per_class = 2 * np.pi / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, -roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
    return ret_box3d


class ProposalLayer(nn.Module):

    def __init__(self, device, nms_pre=9000, nms_post=512, nms_thres=0.85, nms_post_val=None, nms_thres_val=None, mean_size=[1.0], loc_xz_fine=True, loc_scope=3.0, loc_bin_size=0.5, num_head_bin=12, get_y_by_bin=False, get_ry_fine=False, loc_y_scope=0.5, loc_y_bin_size=0.25, post_process=True):
        super().__init__()
        self.nms_pre = nms_pre
        self.nms_post = nms_post
        self.nms_thres = nms_thres
        self.nms_post_val = nms_post_val
        self.nms_thres_val = nms_thres_val
        self.mean_size = torch.tensor(mean_size, device=device)
        self.loc_scope = loc_scope
        self.loc_bin_size = loc_bin_size
        self.num_head_bin = num_head_bin
        self.loc_xz_fine = loc_xz_fine
        self.get_y_by_bin = get_y_by_bin
        self.get_ry_fine = get_ry_fine
        self.loc_y_scope = loc_y_scope
        self.loc_y_bin_size = loc_y_bin_size
        self.post_process = post_process

    def forward(self, rpn_scores, rpn_reg, xyz):
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, xyz.shape[-1]), rpn_reg.view(-1, rpn_reg.shape[-1]), anchor_size=self.mean_size, loc_scope=self.loc_scope, loc_bin_size=self.loc_bin_size, num_head_bin=self.num_head_bin, get_xz_fine=self.loc_xz_fine, get_y_by_bin=self.get_y_by_bin, get_ry_fine=self.get_ry_fine, loc_y_scope=self.loc_y_scope, loc_y_bin_size=self.loc_y_bin_size)
        proposals = proposals.view(batch_size, -1, 7)
        nms_post = self.nms_post
        nms_thres = self.nms_thres
        if not self.training:
            if self.nms_post_val is not None:
                nms_post = self.nms_post_val
            if self.nms_thres_val is not None:
                nms_thres = self.nms_thres_val
        if self.post_process:
            proposals[..., 1] += proposals[..., 3] / 2
            scores = rpn_scores
            _, sorted_idxs = torch.sort(scores, dim=1, descending=True)
            batch_size = scores.size(0)
            ret_bbox3d = scores.new(batch_size, nms_post, 7).zero_()
            ret_scores = scores.new(batch_size, nms_post).zero_()
            for k in range(batch_size):
                scores_single = scores[k]
                proposals_single = proposals[k]
                order_single = sorted_idxs[k]
                scores_single, proposals_single = self.distance_based_proposal(scores_single, proposals_single, order_single)
                proposals_tot = proposals_single.size(0)
                ret_bbox3d[k, :proposals_tot] = proposals_single
                ret_scores[k, :proposals_tot] = scores_single
        else:
            batch_size = rpn_scores.size(0)
            ret_bbox3d = []
            ret_scores = []
            for k in range(batch_size):
                bev = xywhr_to_xyxyr(proposals[k, :, [0, 2, 3, 5, 6]])
                keep_idx = nms(bev, rpn_scores[k], nms_thres)
                ret_bbox3d.append(proposals[k, keep_idx])
                ret_scores.append(rpn_scores[k, keep_idx])
        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """Propose ROIs in two area based on the distance.

        Args:
            scores: (N)
            proposals: (N, 7)
            order: (N)
        """
        nms_post = self.nms_post
        nms_thres = self.nms_thres
        if not self.training:
            if self.nms_post_val is not None:
                nms_post = self.nms_post_val
            if self.nms_thres_val is not None:
                nms_thres = self.nms_thres_val
        nms_range_list = [0, 40.0, 80.0]
        pre_top_n_list = [0, int(self.nms_pre * 0.7), self.nms_pre - int(self.nms_pre * 0.7)]
        post_top_n_list = [0, int(nms_post * 0.7), nms_post - int(nms_post * 0.7)]
        scores_single_list, proposals_single_list = [], []
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            dist_mask = (dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i])
            if dist_mask.sum() != 0:
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
            bev = xywhr_to_xyxyr(cur_proposals[:, [0, 2, 3, 5, 6]])
            keep_idx = nms(bev, cur_scores, nms_thres)
            keep_idx = keep_idx[:post_top_n_list[i]]
            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])
        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single


class ProposalTargetLayer(nn.Module):

    def __init__(self, pool_extra_width=1.0, num_points=512, reg_fg_thresh=0.55, cls_fg_thresh=0.6, cls_bg_thresh=0.45, cls_bg_thresh_lo=0.05, fg_ratio=0.5, roi_per_image=64, aug_rot_range=18, hard_bg_ratio=0.8, roi_fg_aug_times=10):
        super().__init__()
        self.pool_extra_width = pool_extra_width
        self.num_points = num_points
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh = cls_bg_thresh
        self.cls_bg_thresh_lo = cls_bg_thresh_lo
        self.fg_ratio = fg_ratio
        self.roi_per_image = roi_per_image
        self.aug_rot_range = aug_rot_range
        self.hard_bg_ratio = hard_bg_ratio
        self.roi_fg_aug_times = roi_fg_aug_times

    def forward(self, x):
        roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature = x
        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)
        pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, self.pool_extra_width, sampled_pt_num=self.num_points)
        sampled_pts, sampled_features = pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:]
        sampled_pts, batch_rois, batch_gt_of_rois = self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - roi_center.unsqueeze(dim=2)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry
        for k in range(batch_size):
            sampled_pts[k] = rotate_pc_along_y_torch(sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1), roi_ry[k]).squeeze(dim=1)
        valid_mask = pooled_empty_flag == 0
        reg_valid_mask = ((batch_roi_iou > self.reg_fg_thresh) & valid_mask).long()
        batch_cls_label = (batch_roi_iou > self.cls_fg_thresh).long()
        invalid_mask = (batch_roi_iou > self.cls_bg_thresh) & (batch_roi_iou < self.cls_fg_thresh)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1
        output_dict = {'sampled_pts': sampled_pts.view(-1, self.num_points, 3), 'pts_feature': sampled_features.view(-1, self.num_points, sampled_features.shape[3]), 'cls_label': batch_cls_label.view(-1), 'reg_valid_mask': reg_valid_mask.view(-1), 'gt_of_rois': batch_gt_of_rois.view(-1, 7), 'gt_iou': batch_roi_iou.view(-1), 'roi_boxes3d': batch_rois.view(-1, 7)}
        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """Sample ROIs for RCNN.

        Args:
            roi_boxes3d: (B, M, 7)
            gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]

        Returns:
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.size(0)
        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_image))
        batch_rois = gt_boxes3d.new(batch_size, self.roi_per_image, 7).zero_()
        batch_gt_of_rois = gt_boxes3d.new(batch_size, self.roi_per_image, 7).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, self.roi_per_image).zero_()
        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            if cur_gt.__len__() == 0:
                cur_gt = torch.zeros(1, 7)
            iou3d = iou_3d(cur_roi.detach().cpu().numpy()[:, [0, 1, 2, 5, 3, 4, 6]], cur_gt[:, 0:7].detach().cpu().numpy()[:, [0, 1, 2, 5, 3, 4, 6]])
            iou3d = torch.tensor(iou3d, device=cur_roi.device)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)
            fg_inds = torch.nonzero(max_overlaps >= fg_thresh).view(-1)
            easy_bg_inds = torch.nonzero(max_overlaps < self.cls_bg_thresh_lo).view(-1)
            hard_bg_inds = torch.nonzero((max_overlaps < self.cls_bg_thresh) & (max_overlaps >= self.cls_bg_thresh_lo)).view(-1)
            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(self.roi_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = self.roi_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                bg_rois_per_this_image = self.roi_per_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
                fg_rois_per_this_image = 0
            else:
                pdb.set_trace()
                raise NotImplementedError
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois, iou3d_src, aug_times=self.roi_fg_aug_times)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)
            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if self.roi_fg_aug_times > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois, iou3d_src, aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)
            rois = torch.cat(roi_list, dim=0)
            iou_of_rois = torch.cat(roi_iou_list, dim=0)
            gt_of_rois = torch.cat(roi_gt_list, dim=0)
            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois
        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * self.hard_bg_ratio)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]
            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError
        return bg_inds

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        iou_of_rois = torch.zeros(roi_boxes3d.shape[0]).type_as(gt_boxes3d)
        pos_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)
        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k].view(1, 7)
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = aug_box3d.view((1, 7))
                iou3d = iou_3d(aug_box3d.detach().cpu().numpy()[:, [0, 1, 2, 5, 3, 4, 6]], gt_box3d.detach().cpu().numpy()[:, [0, 1, 2, 5, 3, 4, 6]])
                iou3d = torch.tensor(iou3d, device=aug_box3d.device)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def random_aug_box3d(box3d):
        """Random shift, scale, orientation.

        Args:
            box3d: (7) [x, y, z, h, w, l, ry]
        """
        range_config = [[0.2, 0.1, np.pi / 12, 0.7], [0.3, 0.15, np.pi / 12, 0.6], [0.5, 0.15, np.pi / 9, 0.5], [0.8, 0.15, np.pi / 6, 0.3], [1.0, 0.15, np.pi / 3, 0.2]]
        idx = torch.randint(low=0, high=len(range_config), size=(1,))[0].long()
        pos_shift = (torch.rand(3, device=box3d.device) - 0.5) / 0.5 * range_config[idx][0]
        hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / 0.5 * range_config[idx][1] + 1.0
        angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / 0.5 * range_config[idx][2]
        aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
        return aug_box3d

    def data_augmentation(self, pts, rois, gt_of_rois):
        """Data augmentation.

        Args:
            pts: (B, M, 512, 3)
            rois: (B, M. 7)
            gt_of_rois: (B, M, 7)
        """
        batch_size, boxes_num = pts.shape[0], pts.shape[1]
        angles = (torch.rand((batch_size, boxes_num), device=pts.device) - 0.5 / 0.5) * (np.pi / self.aug_rot_range)
        temp_x, temp_z, temp_ry = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2], gt_of_rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        gt_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry
        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        roi_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry
        for k in range(batch_size):
            pts[k] = rotate_pc_along_y_torch(pts[k], angles[k])
            gt_of_rois[k] = rotate_pc_along_y_torch(gt_of_rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)
            rois[k] = rotate_pc_along_y_torch(rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)
        temp_x, temp_z = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2]
        temp_beta = torch.atan2(temp_z, temp_x)
        gt_of_rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + gt_alpha - temp_beta
        temp_x, temp_z = rois[:, :, 0], rois[:, :, 2]
        temp_beta = torch.atan2(temp_z, temp_x)
        rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + roi_alpha - temp_beta
        scales = 1 + (torch.rand((batch_size, boxes_num), device=pts.device) - 0.5) / 0.5 * 0.05
        pts = pts * scales.unsqueeze(dim=2).unsqueeze(dim=3)
        gt_of_rois[:, :, 0:6] = gt_of_rois[:, :, 0:6] * scales.unsqueeze(dim=2)
        rois[:, :, 0:6] = rois[:, :, 0:6] * scales.unsqueeze(dim=2)
        flip_flag = torch.sign(torch.rand((batch_size, boxes_num), device=pts.device) - 0.5)
        pts[:, :, :, 0] = pts[:, :, :, 0] * flip_flag.unsqueeze(dim=2)
        gt_of_rois[:, :, 0] = gt_of_rois[:, :, 0] * flip_flag
        src_ry = gt_of_rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        gt_of_rois[:, :, 6] = ry
        rois[:, :, 0] = rois[:, :, 0] * flip_flag
        src_ry = rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        rois[:, :, 6] = ry
        return pts, rois, gt_of_rois


def gen_CNN(channels, conv=nn.Conv1d, bias=True, activation=nn.ReLU, batch_norm=None, instance_norm=None):
    layers = []
    for i in range(len(channels) - 1):
        in_size, out_size = channels[i:i + 2]
        layers.append(conv(in_size, out_size, 1, bias=bias))
        if batch_norm is not None:
            layers.append(batch_norm(out_size))
        if activation is not None:
            layers.append(activation(inplace=True))
        if instance_norm is not None:
            layers.append(instance_norm(out_size, affine=False, track_running_stats=False))
    return nn.Sequential(*layers)


def get_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size, get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """Bin-based 3D bounding boxes regression loss. See
    https://arxiv.org/abs/1812.04244 for more details.

    Args:
        pred_reg: (N, C)
        reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
        loc_scope: constant
        loc_bin_size: constant
        num_head_bin: constant
        anchor_size: (N, 3) or (3)
        get_xz_fine: bool
        get_y_by_bin: bool
        loc_y_scope: float
        loc_y_bin_size: float
        get_ry_fine: bool
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2
    reg_loss_dict = {}
    loc_loss = 0
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 0.001)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 0.001)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r
    loss_x_bin = CrossEntropyLoss()(pred_reg[:, x_bin_l:x_bin_r], x_bin_label)
    loss_z_bin = CrossEntropyLoss()(pred_reg[:, z_bin_l:z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r
        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size
        x_bin_onehot = torch.zeros((x_bin_label.size(0), per_loc_bin_num), device=anchor_size.device, dtype=torch.float32)
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.zeros((z_bin_label.size(0), per_loc_bin_num), device=anchor_size.device, dtype=torch.float32)
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)
        loss_x_res = SmoothL1Loss()((pred_reg[:, x_res_l:x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = SmoothL1Loss()((pred_reg[:, z_res_l:z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r
        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 0.001)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size
        y_bin_onehot = one_hot(y_bin_label, loc_y_bin_num)
        loss_y_bin = CrossEntropyLoss()(pred_reg[:, y_bin_l:y_bin_r], y_bin_label)
        loss_y_res = SmoothL1Loss()((pred_reg[:, y_res_l:y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)
        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()
        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r
        loss_y_offset = SmoothL1Loss()(pred_reg[:, y_offset_l:y_offset_r].sum(dim=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    ry_label = reg_label[:, 6]
    if get_ry_fine:
        angle_per_class = np.pi / 2 / num_head_bin
        ry_label = ry_label % (2 * np.pi)
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)
        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=0.001, max=np.pi * 0.5 - 0.001)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)
    else:
        angle_per_class = 2 * np.pi / num_head_bin
        heading_angle = ry_label % (2 * np.pi)
        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)
    ry_bin_onehot = one_hot(ry_bin_label, num_head_bin)
    loss_ry_bin = CrossEntropyLoss()(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = SmoothL1Loss()((pred_reg[:, ry_res_l:ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)
    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)
    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = SmoothL1Loss()(size_res_norm, size_res_norm_label)
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss
    return loc_loss, angle_loss, size_loss, reg_loss_dict


class RCNN(nn.Module):

    def __init__(self, num_classes, device, in_channels=128, SA_config={'npoints': [128, 32, -1], 'radius': [0.2, 0.4, 100], 'nsample': [64, 64, 64], 'mlps': [[128, 128, 128], [128, 128, 256], [256, 256, 512]]}, cls_out_ch=[256, 256], reg_out_ch=[256, 256], db_ratio=0.5, use_xyz=True, xyz_up_layer=[128, 128], head={}, target_head={}, loss={}):
        super().__init__()
        self.rcnn_input_channel = 5
        self.pool_extra_width = target_head.get('pool_extra_width', 1.0)
        self.num_points = target_head.get('num_points', 512)
        self.proposal_layer = ProposalLayer(device=device, **head)
        self.SA_modules = nn.ModuleList()
        for i in range(len(SA_config['npoints'])):
            mlps = [in_channels] + SA_config['mlps'][i]
            npoint = SA_config['npoints'][i] if SA_config['npoints'][i] != -1 else None
            self.SA_modules.append(PointnetSAModule(npoint=npoint, radius=SA_config['radius'][i], nsample=SA_config['nsample'][i], mlp=mlps, use_xyz=use_xyz, bias=True))
            in_channels = mlps[-1]
        self.xyz_up_layer = gen_CNN([self.rcnn_input_channel] + xyz_up_layer, conv=nn.Conv2d)
        c_out = xyz_up_layer[-1]
        self.merge_down_layer = gen_CNN([c_out * 2, c_out], conv=nn.Conv2d)
        cls_channel = 1 if num_classes == 2 else num_classes
        in_filters = [in_channels, *cls_out_ch[:-1]]
        layers = []
        for i in range(len(cls_out_ch)):
            layers.extend([nn.Conv1d(in_filters[i], cls_out_ch[i], 1, bias=True), nn.ReLU(inplace=True)])
        layers.append(nn.Conv1d(cls_out_ch[-1], cls_channel, 1, bias=True))
        self.cls_blocks = nn.Sequential(*layers)
        self.loss_cls = nn.functional.binary_cross_entropy
        per_loc_bin_num = int(self.proposal_layer.loc_scope / self.proposal_layer.loc_bin_size) * 2
        loc_y_bin_num = int(self.proposal_layer.loc_y_scope / self.proposal_layer.loc_y_bin_size) * 2
        reg_channel = per_loc_bin_num * 4 + self.proposal_layer.num_head_bin * 2 + 3
        reg_channel += 1 if not self.proposal_layer.get_y_by_bin else loc_y_bin_num * 2
        in_filters = [in_channels, *reg_out_ch[:-1]]
        layers = []
        for i in range(len(reg_out_ch)):
            layers.extend([nn.Conv1d(in_filters[i], reg_out_ch[i], 1, bias=True), nn.ReLU(inplace=True)])
        layers.append(nn.Conv1d(reg_out_ch[-1], reg_channel, 1, bias=True))
        self.reg_blocks = nn.Sequential(*layers)
        self.proposal_target_layer = ProposalTargetLayer(**target_head)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_blocks[-1].weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, roi_boxes3d, gt_boxes3d, rpn_xyz, rpn_features, seg_mask, pts_depth):
        pts_extra_input_list = [seg_mask.unsqueeze(dim=2)]
        pts_extra_input_list.append((pts_depth / 70.0 - 0.5).unsqueeze(dim=2))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=2)
        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
        if gt_boxes3d[0] is not None:
            max_gt = 0
            for bbox in gt_boxes3d:
                max_gt = max(max_gt, bbox.shape[0])
            pad_bboxes = torch.zeros((len(gt_boxes3d), max_gt, 7), dtype=torch.float32, device=gt_boxes3d[0].device)
            for i in range(len(gt_boxes3d)):
                pad_bboxes[i, :gt_boxes3d[i].shape[0], :] = gt_boxes3d[i]
            gt_boxes3d = pad_bboxes
            with torch.no_grad():
                target = self.proposal_target_layer([roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature])
            pts_input = torch.cat((target['sampled_pts'], target['pts_feature']), dim=2)
            target['pts_input'] = pts_input
        else:
            pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, roi_boxes3d, self.pool_extra_width, sampled_pt_num=self.num_points)
            batch_size = roi_boxes3d.shape[0]
            roi_center = roi_boxes3d[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            for k in range(batch_size):
                pooled_features[k, :, :, 0:3] = rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3], roi_boxes3d[k, :, 6])
            pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        xyz, features = self._break_up_pc(pts_input)
        xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
        xyz_feature = self.xyz_up_layer(xyz_input)
        rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)
        merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
        merged_feature = self.merge_down_layer(merged_feature)
        l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        rcnn_cls = self.cls_blocks(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_blocks(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        ret_dict = {'rois': roi_boxes3d, 'cls': rcnn_cls, 'reg': rcnn_reg}
        if gt_boxes3d[0] is not None:
            ret_dict.update(target)
        return ret_dict

    def loss(self, results, inputs):
        rcnn_cls = results['cls']
        rcnn_reg = results['reg']
        cls_label = results['cls_label'].float()
        reg_valid_mask = results['reg_valid_mask']
        roi_boxes3d = results['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = results['gt_of_rois']
        pts_input = results['pts_input']
        cls_label_flat = cls_label.view(-1)
        rcnn_cls_flat = rcnn_cls.view(-1)
        batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
        cls_valid_mask = (cls_label_flat >= 0).float()
        rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        batch_size = pts_input.shape[0]
        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            anchor_size = self.proposal_layer.mean_size
            loss_loc, loss_angle, loss_size, _ = get_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask], gt_boxes3d_ct.view(batch_size, 7)[fg_mask], loc_scope=self.proposal_layer.loc_scope, loc_bin_size=self.proposal_layer.loc_bin_size, num_head_bin=self.proposal_layer.num_head_bin, anchor_size=anchor_size, get_xz_fine=True, get_y_by_bin=self.proposal_layer.get_y_by_bin, loc_y_scope=self.proposal_layer.loc_y_scope, loc_y_bin_size=self.proposal_layer.loc_y_bin_size, get_ry_fine=True)
            loss_size = 3 * loss_size
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            rcnn_loss_reg = rcnn_loss_cls * 0
        return {'cls': rcnn_loss_cls, 'reg': rcnn_loss_reg}


class PointnetFPModule(nn.Module):
    """Propagates the features of one set to another."""

    def __init__(self, *, mlp: List[int], bn: bool=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) ->torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn_gpu(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate_gpu(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class Pointnet2MSG(nn.Module):

    def __init__(self, in_channels=6, use_xyz=True, SA_config={'npoints': [128, 32, -1], 'radius': [0.2, 0.4, 100], 'nsample': [64, 64, 64], 'mlps': [[128, 128, 128], [128, 128, 256], [256, 256, 512]]}, fp_mlps=[]):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        skip_channel_list = [in_channels]
        for i in range(len(SA_config['npoints'])):
            mlps = SA_config['mlps'][i].copy()
            out_channels = 0
            for idx in range(len(mlps)):
                mlps[idx] = [in_channels] + mlps[idx]
                out_channels += mlps[idx][-1]
            self.SA_modules.append(PointnetSAModuleMSG(npoint=SA_config['npoints'][i], radii=SA_config['radius'][i], nsamples=SA_config['nsample'][i], mlps=mlps, use_xyz=use_xyz, batch_norm=nn.BatchNorm2d))
            in_channels = out_channels
            skip_channel_list.append(out_channels)
        self.FP_modules = nn.ModuleList()
        for i in range(len(fp_mlps)):
            pre_channel = fp_mlps[i + 1][-1] if i + 1 < len(fp_mlps) else out_channels
            self.FP_modules.append(PointnetFPModule(mlp=[pre_channel + skip_channel_list[i]] + fp_mlps[i], batch_norm=nn.BatchNorm2d))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        return l_xyz[0], l_features[0]


class RPN(nn.Module):

    def __init__(self, device, backbone={}, cls_in_ch=128, cls_out_ch=[128], reg_in_ch=128, reg_out_ch=[128], db_ratio=0.5, head={}, focal_loss={}, loss_weight=[1.0, 1.0], **kwargs):
        super().__init__()
        self.backbone = Pointnet2MSG(**backbone)
        self.proposal_layer = ProposalLayer(device=device, **head)
        in_filters = [cls_in_ch, *cls_out_ch[:-1]]
        layers = []
        for i in range(len(cls_out_ch)):
            layers.extend([nn.Conv1d(in_filters[i], cls_out_ch[i], 1, bias=False), nn.BatchNorm1d(cls_out_ch[i]), nn.ReLU(inplace=True), nn.Dropout(db_ratio)])
        layers.append(nn.Conv1d(cls_out_ch[-1], 1, 1, bias=True))
        self.cls_blocks = nn.Sequential(*layers)
        per_loc_bin_num = int(self.proposal_layer.loc_scope / self.proposal_layer.loc_bin_size) * 2
        if self.proposal_layer.loc_xz_fine:
            reg_channel = per_loc_bin_num * 4 + self.proposal_layer.num_head_bin * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + self.proposal_layer.num_head_bin * 2 + 3
        reg_channel = reg_channel + 1
        in_filters = [reg_in_ch, *reg_out_ch[:-1]]
        layers = []
        for i in range(len(reg_out_ch)):
            layers.extend([nn.Conv1d(in_filters[i], reg_out_ch[i], 1, bias=False), nn.BatchNorm1d(reg_out_ch[i]), nn.ReLU(inplace=True), nn.Dropout(db_ratio)])
        layers.append(nn.Conv1d(reg_out_ch[-1], reg_channel, 1, bias=True))
        self.reg_blocks = nn.Sequential(*layers)
        self.loss_cls = FocalLoss(**focal_loss)
        self.loss_weight = loss_weight
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_blocks[-1].bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_blocks[-1].weight, mean=0, std=0.001)

    def forward(self, x):
        backbone_xyz, backbone_features = self.backbone(x)
        rpn_cls = self.cls_blocks(backbone_features).transpose(1, 2).contiguous()
        rpn_reg = self.reg_blocks(backbone_features).transpose(1, 2).contiguous()
        return rpn_cls, rpn_reg, backbone_xyz, backbone_features

    def loss(self, results, inputs):
        rpn_cls = results['cls']
        rpn_reg = results['reg']
        rpn_cls_label = torch.stack(inputs.labels)
        rpn_reg_label = torch.stack(inputs.bboxes)
        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = rpn_cls_label_flat > 0
        rpn_cls_target = (rpn_cls_label_flat > 0).int()
        pos = (rpn_cls_label_flat > 0).float()
        neg = (rpn_cls_label_flat == 0).float()
        cls_weights = pos + neg
        pos_normalizer = pos.sum()
        cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
        rpn_loss_cls = self.loss_cls(rpn_cls_flat, rpn_cls_target, cls_weights, avg_factor=1.0)
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask], rpn_reg_label.view(point_num, 7)[fg_mask], loc_scope=self.proposal_layer.loc_scope, loc_bin_size=self.proposal_layer.loc_bin_size, num_head_bin=self.proposal_layer.num_head_bin, anchor_size=self.proposal_layer.mean_size, get_xz_fine=self.proposal_layer.loc_xz_fine, get_y_by_bin=False, get_ry_fine=False)
            loss_size = 3 * loss_size
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            rpn_loss_reg = rpn_loss_cls * 0
        return {'cls': rpn_loss_cls * self.loss_weight[0], 'reg': rpn_loss_reg * self.loss_weight[1]}


class PointRCNN(BaseModel):
    """Object detection model. Based on the PoinRCNN architecture
    https://github.com/sshaoshuai/PointRCNN.

    The network is not trainable end-to-end, it requires pre-training of the RPN
    module, followed by training of the RCNN module.  For this the mode must be
    set to 'RPN', with this, the network only outputs intermediate results.  If
    the RPN module is trained, the mode can be set to 'RCNN' (default), with
    this, the second module can be trained and the output are the final
    predictions.

    For inference use the 'RCNN' mode.

    Args:
        name (string): Name of model.
            Default to "PointRCNN".
        device (string): 'cuda' or 'cpu'.
            Default to 'cuda'.
        classes (string[]): List of classes used for object detection:
            Default to ['Car'].
        score_thres (float): Min confindence score for prediction.
            Default to 0.3.
        npoints (int): Number of processed input points.
            Default to 16384.
        rpn (dict): Config of RPN module.
            Default to {}.
        rcnn (dict): Config of RCNN module.
            Default to {}.
        mode (string): Execution mode, 'RPN' or 'RCNN'.
            Default to 'RCNN'.
    """

    def __init__(self, name='PointRCNN', device='cuda', classes=['Car'], score_thres=0.3, npoints=16384, rpn={}, rcnn={}, mode='RCNN', **kwargs):
        super().__init__(name=name, device=device, **kwargs)
        assert mode == 'RPN' or mode == 'RCNN'
        self.mode = mode
        self.augmenter = ObjdetAugmentation(self.cfg.augment, seed=self.rng)
        self.npoints = npoints
        self.classes = classes
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.lbl2name = {i: n for i, n in enumerate(classes)}
        self.score_thres = score_thres
        self.rpn = RPN(device=device, **rpn)
        self.rcnn = RCNN(device=device, num_classes=len(self.classes), **rcnn)
        self.device = device
        self

    def forward(self, inputs):
        points = torch.stack(inputs.point)
        with torch.set_grad_enabled(self.training and self.mode == 'RPN'):
            if not self.mode == 'RPN':
                self.rpn.eval()
            cls_score, reg_score, backbone_xyz, backbone_features = self.rpn(points)
            with torch.no_grad():
                rpn_scores_raw = cls_score[:, :, 0]
                rois, _ = self.rpn.proposal_layer(rpn_scores_raw, reg_score, backbone_xyz)
            output = {'rois': rois, 'cls': cls_score, 'reg': reg_score}
        if self.mode == 'RCNN':
            with torch.no_grad():
                rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > self.score_thres).float()
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)
            output = self.rcnn(rois, inputs.bboxes, backbone_xyz, backbone_features.permute((0, 2, 1)), seg_mask, pts_depth)
        return output

    def get_optimizer(self, cfg):

        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) ->int:
            return len(children(m))
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        optimizer_func = partial(torch.optim.Adam, betas=tuple(cfg.betas))
        optimizer = OptimWrapper.create(optimizer_func, 0.003, get_layer_groups(self), wd=cfg.weight_decay, true_wd=True, bn_wd=True)
        if self.mode == 'RCNN':
            for param in self.rpn.parameters():
                param.requires_grad = False
        lr_scheduler = OneCycleScheduler(optimizer, 40800, cfg.lr, list(cfg.moms), cfg.div_factor, cfg.pct_start)


        class CustomScheduler:

            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.it = 0

            def step(self):
                self.it += 3000
                self.scheduler.step(self.it)
        scheduler = CustomScheduler(lr_scheduler)
        return optimizer, scheduler

    def get_loss(self, results, inputs):
        if self.mode == 'RPN':
            return self.rpn.loss(results, inputs)
        else:
            if not self.training:
                return {}
            return self.rcnn.loss(results, inputs)

    def filter_objects(self, bbox_objs):
        """Filter objects based on classes to train.

        Args:
            bbox_objs: Bounding box objects from dataset class.

        Returns:
            Filtered bounding box objects.

        """
        filtered = []
        for bb in bbox_objs:
            if bb.label_class in self.classes:
                filtered.append(bb)
        return filtered

    def preprocess(self, data, attr):
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng
        if attr['split'] in ['train', 'training']:
            data = self.augmenter.augment(data, attr, seed=rng)
        data['bounding_boxes'] = self.filter_objects(data['bounding_boxes'])
        points = np.array(data['point'][..., :3], dtype=np.float32)
        calib = data['calib']
        points = DataProcessing.world2cam(points, calib['world_cam'])
        new_data = {'point': points, 'calib': calib}
        if attr['split'] not in ['test', 'testing']:
            new_data['bbox_objs'] = data['bounding_boxes']
        return new_data

    @staticmethod
    def generate_rpn_training_labels(points, bboxes, bboxes_world, calib=None):
        """Generates labels for RPN network.

        Classifies each point as foreground/background based on points inside bbox.
        We don't train on ambiguous points which are just outside bounding boxes(calculated
        by `extended_boxes`).
        Also computes regression labels for bounding box proposals(in bounding box frame).

        Args:
            points: Input pointcloud.
            bboxes: bounding boxes in camera frame.
            bboxes_world: bounding boxes in world frame.
            calib: Calibration file for cam_to_world matrix.

        Returns:
            Classification and Regression labels.

        """
        cls_label = np.zeros(points.shape[0], dtype=np.int32)
        reg_label = np.zeros((points.shape[0], 7), dtype=np.float32)
        if len(bboxes) == 0:
            return cls_label, reg_label
        pts_idx = points_in_box(points.copy(), bboxes_world, camera_frame=True, cam_world=DataProcessing.invT(calib['world_cam']))
        extended_boxes = bboxes_world.copy()
        extended_boxes[3:6] += 0.4
        extended_boxes[:, 2] -= 0.2
        pts_idx_ext = points_in_box(points.copy(), extended_boxes, camera_frame=True, cam_world=DataProcessing.invT(calib['world_cam']))
        for k in range(bboxes.shape[0]):
            fg_pt_flag = pts_idx[:, k]
            fg_pts_rect = points[fg_pt_flag]
            cls_label[fg_pt_flag] = 1
            fg_enlarge_flag = pts_idx_ext[:, k]
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1
            center3d = bboxes[k][0:3].copy()
            center3d[1] -= bboxes[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect
            reg_label[fg_pt_flag, 3] = bboxes[k][3]
            reg_label[fg_pt_flag, 4] = bboxes[k][4]
            reg_label[fg_pt_flag, 5] = bboxes[k][5]
            reg_label[fg_pt_flag, 6] = bboxes[k][6]
        return cls_label, reg_label

    def transform(self, data, attr):
        points = data['point']
        if attr['split'] not in ['test', 'testing']:
            if torch.utils.data.get_worker_info():
                seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
                rng = np.random.default_rng(seedseq.spawn(1)[0])
            else:
                rng = self.rng
            if self.npoints < len(points):
                pts_depth = points[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = rng.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) if len(far_idxs_choice) > 0 else near_idxs_choice
                rng.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                if self.npoints > len(points):
                    extra_choice = rng.choice(choice, self.npoints - len(points), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                rng.shuffle(choice)
            points = points[choice, :]
        t_data = {'point': points, 'calib': data['calib']}
        if attr['split'] not in ['test', 'testing']:
            labels = []
            bboxes = []
            bboxes_world = []
            if len(data['bbox_objs']) != 0:
                labels = np.stack([self.name2lbl.get(bb.label_class, len(self.classes)) for bb in data['bbox_objs']])
                bboxes = np.stack([bb.to_camera() for bb in data['bbox_objs']])
                bboxes_world = np.stack([bb.to_xyzwhlr() for bb in data['bbox_objs']])
            if self.mode == 'RPN':
                labels, bboxes = PointRCNN.generate_rpn_training_labels(points, bboxes, bboxes_world, data['calib'])
            t_data['labels'] = labels
            t_data['bbox_objs'] = data['bbox_objs']
            if attr['split'] in ['train', 'training'] or self.mode == 'RPN':
                t_data['bboxes'] = bboxes
        return t_data

    def inference_end(self, results, inputs):
        if self.mode == 'RPN':
            return [[]]
        roi_boxes3d = results['rois']
        batch_size = roi_boxes3d.shape[0]
        rcnn_cls = results['cls'].view(batch_size, -1, results['cls'].shape[1])
        rcnn_reg = results['reg'].view(batch_size, -1, results['reg'].shape[1])
        pred_boxes3d, rcnn_cls = self.rcnn.proposal_layer(rcnn_cls, rcnn_reg, roi_boxes3d)
        inference_result = []
        for calib, bboxes, scores in zip(inputs.calib, pred_boxes3d, rcnn_cls):
            if scores.shape[-1] == 1:
                scores = torch.sigmoid(scores)
                labels = (scores < self.score_thres).long()
            else:
                labels = torch.argmax(scores)
                scores = F.softmax(scores, dim=0)
                scores = scores[labels]
            fltr = torch.flatten(scores > self.score_thres)
            bboxes = bboxes[fltr]
            labels = labels[fltr]
            scores = scores[fltr]
            bboxes = bboxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            inference_result.append([])
            world_cam, cam_img = None, None
            if calib is not None:
                world_cam = calib.get('world_cam', None)
                cam_img = calib.get('cam_img', None)
            for bbox, score, label in zip(bboxes, scores, labels):
                pos = bbox[:3]
                dim = bbox[[4, 3, 5]]
                pos = DataProcessing.cam2world(pos.reshape((1, -1)), world_cam).flatten()
                pos = pos + [0, 0, dim[1] / 2]
                yaw = bbox[-1]
                name = self.lbl2name.get(label[0], 'ignore')
                inference_result[-1].append(BEVBox3D(pos, dim, yaw, name, score, world_cam, cam_img))
        return inference_result


def knn_batch(points, queries, k, points_row_splits, queries_row_splits, return_distances=True):
    """K nearest neighbour with batch support.

    Args:
        points: Input pointcloud.
        queries: Queries for Knn.
        k: Number of neighbours.
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        return_distances: Whether to return distance with neighbours.

    """
    if points_row_splits.shape[0] != queries_row_splits.shape[0]:
        raise ValueError('KNN(points and queries must have same batch size)')
    points = points.cpu()
    queries = queries.cpu()
    ans = knn_search(points, queries, k=k, points_row_splits=points_row_splits, queries_row_splits=queries_row_splits, return_distances=True)
    if return_distances:
        return ans.neighbors_index.reshape(-1, k).long(), ans.neighbors_distance.reshape(-1, k)
    else:
        return ans.neighbors_index.reshape(-1, k).long()


def queryandgroup(nsample, points, queries, feat, idx, points_row_splits, queries_row_splits, use_xyz=True):
    """Find nearest neighbours and returns grouped features.

    Args:
        nsample: Number of neighbours (k).
        points: Input pointcloud (n, 3).
        queries: Queries for Knn (m, 3).
        feat: features (n, c).
        idx: Optional knn index list.
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        use_xyz: Whether to return xyz concatenated with features.

    Returns:
        Returns grouped features (m, nsample, c) or (m, nsample, 3+c).

    """
    if not (points.is_contiguous and queries.is_contiguous() and feat.is_contiguous()):
        raise ValueError('queryandgroup (points/queries/feat not contiguous)')
    if queries is None:
        queries = points
    if idx is None:
        idx = knn_batch(points, queries, k=nsample, points_row_splits=points_row_splits, queries_row_splits=queries_row_splits, return_distances=False)
    n, m, c = points.shape[0], queries.shape[0], feat.shape[1]
    grouped_xyz = points[idx.view(-1).long(), :].view(m, nsample, 3)
    grouped_xyz -= queries.unsqueeze(1)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)
    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat


class Transformer(nn.Module):
    """Transformer layer of the model, uses self attention."""

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        """Constructor for Transformer Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.

        """
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Linear(mid_planes, mid_planes // share_planes), nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True), nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        """Forward call for Transformer.

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            Transformer features.

        """
        point, feat, row_splits = pxo
        feat_q, feat_k, feat_v = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        feat_k = queryandgroup(self.nsample, point, point, feat_k, None, row_splits, row_splits, use_xyz=True)
        feat_v = queryandgroup(self.nsample, point, point, feat_v, None, row_splits, row_splits, use_xyz=False)
        point_r, feat_k = feat_k[:, :, 0:3], feat_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            point_r = layer(point_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(point_r)
        w = feat_k - feat_q.unsqueeze(1) + point_r.view(point_r.shape[0], point_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)
        n, nsample, c = feat_v.shape
        s = self.share_planes
        feat = ((feat_v + point_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return feat


class Bottleneck(nn.Module):
    """Bottleneck layer for PointTransformer.

    Block of layers using Transformer layer as building block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        """Constructor for Bottleneck Layer.

        Args:
            in_planes (int): Number of input planes.
            planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.

        """
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """Forward call for Bottleneck

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo
        identity = feat
        feat = self.relu(self.bn1(self.linear1(feat)))
        feat = self.relu(self.bn2(self.transformer2([point, feat, row_splits])))
        feat = self.bn3(self.linear3(feat))
        feat += identity
        feat = self.relu(feat)
        return [point, feat, row_splits]


class SemsegAugmentation(Augmentation):
    """Class consisting of different augmentation methods for Semantic Segmentation.

    Args:
        cfg: Config for augmentation.
    """

    def __init__(self, cfg, seed=None):
        super(SemsegAugmentation, self).__init__(cfg, seed=seed)
        all_methods = ['recenter', 'normalize', 'rotate', 'scale', 'noise', 'RandomDropout', 'RandomHorizontalFlip', 'ChromaticAutoContrast', 'ChromaticTranslation', 'ChromaticJitter', 'HueSaturationTranslation']
        if cfg is None:
            return
        for method in cfg:
            if method not in all_methods:
                warnings.warn(f'Augmentation method : {method} does not exist. Please verify!')

    def RandomDropout(self, pc, feats, labels, cfg):
        """Randomly drops some points.

        Args:
            pc: Pointcloud.
            feats: Features.
            labels: Labels.
            cfg: configuration dict.
        """
        dropout_ratio = cfg.get('dropout_ratio', 0.2)
        if self.rng.random() < dropout_ratio:
            N = len(pc)
            inds = self.rng.choice(N, int(N * (1 - dropout_ratio)), replace=False)
            return pc[inds], feats[inds], labels[inds]
        return pc, feats, labels

    def RandomHorizontalFlip(self, pc, cfg):
        """Randomly flips the given axes.

        Args:
            pc: Pointcloud.
            cfg: configuraiton dict.

        """
        axes = cfg.get('axes', [0, 1])
        if self.rng.random() < 0.95:
            for curr_ax in axes:
                if self.rng.random() < 0.5:
                    pc_max = np.max(pc[:, curr_ax])
                    pc[:, curr_ax] = pc_max - pc[:, curr_ax]
        return pc

    def ChromaticAutoContrast(self, feats, cfg):
        """Improve contrast for RGB features.

        Args:
            feats: RGB features, should be in range [0-255].
            cfg: configuration dict.

        """
        randomize_blend_factor = cfg.get('randomize_blend_factor', True)
        blend_factor = cfg.get('blend_factor', 0.5)
        if self.rng.random() < 0.2:
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, 'Invalid color value. Color is supposed to be in [0-255] for ChromaticAutoContrast augmentation'
            scale = 255 / (hi - lo)
            contrast_feats = (feats[:, :3] - lo) * scale
            blend_factor = self.rng.random() if randomize_blend_factor else blend_factor
            feats[:, :3] = (1 - blend_factor) * feats[:, :3] + blend_factor * contrast_feats
        return feats

    def ChromaticTranslation(self, feats, cfg):
        """Adds a small translation vector to features.

        Args:
            feats: Features.
            cfg: configuration dict.

        """
        trans_range_ratio = cfg.get('trans_range_ratio', 0.1)
        if self.rng.random() < 0.95:
            tr = (self.rng.random((1, 3)) - 0.5) * 255 * 2 * trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return feats

    def ChromaticJitter(self, feats, cfg):
        """Adds a small noise jitter to features.

        Args:
            feats: Features.
            cfg: configuration dict.

        """
        std = cfg.get('std', 0.01)
        if self.rng.random() < 0.95:
            noise = self.rng.standard_normal((feats.shape[0], 3))
            noise *= std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return feats

    @staticmethod
    def _rgb_to_hsv(rgb):
        """Converts RGB to HSV.

        Translated from source of colorsys.rgb_to_hsv
        r,g,b should be a numpy arrays with values between 0 and 255
        rgb_to_hsv returns an array of floats between 0.0 and 1.0.

        Args:
            rgb: RGB image

        Returns:
            HSV image

        """
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = hsv[..., 0] / 6.0 % 1.0
        return hsv

    @staticmethod
    def _hsv_to_rgb(hsv):
        """Converts HSV to RGB.

        Translated from source of colorsys.hsv_to_rgb
        h,s should be a numpy arrays with values between 0.0 and 1.0
        v should be a numpy array with values between 0.0 and 255.0
        hsv_to_rgb returns an array of uints between 0 and 255.

        Args:
            hsv: HSV image

        Returns:
            RGB image

        """
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    @staticmethod
    def HueSaturationTranslation(feat, cfg):
        """Adds small noise to hue and saturation.

        Args:
            feat: Features.
            cfg: config dict with keys('hue_max', and 'saturation_max').

        """
        hue_max = cfg.get('hue_max', 0.5)
        saturation_max = cfg.get('saturation_max', 0.2)
        hsv = SemsegAugmentation._rgb_to_hsv(feat[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(SemsegAugmentation._hsv_to_rgb(hsv), 0, 255)
        return feat

    def augment(self, point, feat, labels, cfg, seed=None):
        if cfg is None:
            return point, feat, labels
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if 'recenter' in cfg:
            point = self.recenter(point, cfg['recenter'])
        if 'normalize' in cfg:
            point, feat = self.normalize(point, feat, cfg['normalize'])
        if 'rotate' in cfg:
            point = self.rotate(point, cfg['rotate'])
        if 'scale' in cfg:
            point = self.scale(point, cfg['scale'])
        if 'noise' in cfg:
            point = self.noise(point, cfg['noise'])
        if 'RandomDropout' in cfg:
            point, feat, labels = self.RandomDropout(point, feat, labels, cfg['RandomDropout'])
        if 'RandomHorizontalFlip' in cfg:
            point = self.RandomHorizontalFlip(point, cfg['RandomHorizontalFlip'])
        if 'ChromaticAutoContrast' in cfg:
            feat = self.ChromaticAutoContrast(feat, cfg['ChromaticAutoContrast'])
        if 'ChromaticTranslation' in cfg:
            feat = self.ChromaticTranslation(feat, cfg['ChromaticTranslation'])
        if 'ChromaticJitter' in cfg:
            feat = self.ChromaticJitter(feat, cfg['ChromaticJitter'])
        if 'HueSaturationTranslation' in cfg:
            feat = self.HueSaturationTranslation(feat, cfg['HueSaturationTranslation'])
        return point, feat, labels


class FurthestPointSamplingV2(Function):
    """Furthest Point Sampling with variable length batch support."""

    @staticmethod
    def forward(ctx, xyz, row_splits, new_row_splits):
        """Forward pass.

        Args:
            ctx: Context.
            xyz (torch.float32): Input pointcloud (N, 3).
            row_splits (torch,int64): splits to define batch (b + 1,)
            new_row_splits (torch.int64): splits for output batch lengths (b + 1,)

        Returns:
            Returns indices of sampled points with shape (new_row_splits[-1], ).
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError
        if not xyz.is_contiguous():
            raise ValueError('FurthestPointSampling : coordinates are not contiguous.')
        idx = []
        for i in range(0, row_splits.shape[0] - 1):
            npoint = new_row_splits[i + 1] - new_row_splits[i]
            start_i = row_splits[i]
            end_i = row_splits[i + 1]
            out = furthest_point_sampling(xyz[start_i:end_i].unsqueeze(0), npoint) + row_splits[i]
            idx += out
        return torch.cat(idx, 0)

    @staticmethod
    def backward(xyz, a=None, b=None):
        return None, None, None


furthest_point_sample_v2 = FurthestPointSamplingV2.apply


class TransitionDown(nn.Module):
    """TransitionDown layer for PointTransformer.

    Subsamples points and increase receptive field.
    """

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        """Constructor for TransitionDown Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            stride (int): subsampling factor.
            nsample (int): Number of neighbours.

        """
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """Forward call for TransitionDown

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo
        if self.stride != 1:
            new_row_splits = [0]
            count = 0
            for i in range(1, row_splits.shape[0]):
                count += (row_splits[i].item() - row_splits[i - 1].item()) // self.stride
                new_row_splits.append(count)
            new_row_splits = torch.LongTensor(new_row_splits)
            idx = furthest_point_sample_v2(point, row_splits, new_row_splits)
            new_point = point[idx.long(), :]
            feat = queryandgroup(self.nsample, point, new_point, feat, None, row_splits, new_row_splits, use_xyz=True)
            feat = self.relu(self.bn(self.linear(feat).transpose(1, 2).contiguous()))
            feat = self.pool(feat).squeeze(-1)
            point, row_splits = new_point, new_row_splits
        else:
            feat = self.relu(self.bn(self.linear(feat)))
        return [point, feat, row_splits]


def interpolation(points, queries, feat, points_row_splits, queries_row_splits, k=3):
    """Interpolation of features with nearest neighbours.

    Args:
        points: Input pointcloud (m, 3).
        queries: Queries for Knn (n, 3).
        feat: features (m, c).
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        k: Number of neighbours.

    Returns:
        Returns interpolated features (n, c).
    """
    if not (points.is_contiguous and queries.is_contiguous() and feat.is_contiguous()):
        raise ValueError('Interpolation (points/queries/feat not contiguous)')
    idx, dist = knn_batch(points, queries, k=k, points_row_splits=points_row_splits, queries_row_splits=queries_row_splits, return_distances=True)
    idx, dist = idx.reshape(-1, k), dist.reshape(-1, k)
    dist_recip = 1.0 / (dist + 1e-08)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    new_feat = torch.FloatTensor(queries.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class TransitionUp(nn.Module):
    """Decoder layer for PointTransformer.

    Interpolate points based on corresponding encoder layer.
    """

    def __init__(self, in_planes, out_planes=None):
        """Constructor for TransitionUp Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.

        """
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        """Forward call for TransitionUp

        Args:
            pxo1: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            pxo2: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            Interpolated features.

        """
        if pxo2 is None:
            _, feat, row_splits = pxo1
            feat_tmp = []
            for i in range(0, row_splits.shape[0] - 1):
                start_i, end_i, count = row_splits[i], row_splits[i + 1], row_splits[i + 1] - row_splits[i]
                feat_b = feat[start_i:end_i, :]
                feat_b = torch.cat((feat_b, self.linear2(feat_b.sum(0, True) / count).repeat(count, 1)), 1)
                feat_tmp.append(feat_b)
            feat = torch.cat(feat_tmp, 0)
            feat = self.linear1(feat)
        else:
            point_1, feat_1, row_splits_1 = pxo1
            point_2, feat_2, row_splits_2 = pxo2
            feat = self.linear1(feat_1) + interpolation(point_2, point_1, self.linear2(feat_2), row_splits_2, row_splits_1)
        return feat


class PointTransformer(BaseModel):
    """Semantic Segmentation model. Based on PointTransformer architecture
    https://arxiv.org/pdf/2012.09164.pdf

    Uses Encoder-Decoder architecture with Transformer layers.

    Attributes:
        name: Name of model.
          Default to "PointTransformer".
        blocks: Number of Bottleneck layers.
        in_channels: Number of features(default 6).
        num_classes: Number of classes.
        voxel_size: Voxel length for subsampling.
        max_voxels: Maximum number of voxels.
        batcher: Batching method for dataloader.
        augment: dictionary for augmentation.
    """

    def __init__(self, name='PointTransformer', blocks=[2, 2, 2, 2, 2], in_channels=6, num_classes=13, voxel_size=0.04, max_voxels=80000, batcher='ConcatBatcher', augment=None, **kwargs):
        super(PointTransformer, self).__init__(name=name, blocks=blocks, in_channels=in_channels, num_classes=num_classes, voxel_size=voxel_size, max_voxels=max_voxels, batcher=batcher, augment=augment, **kwargs)
        cfg = self.cfg
        self.in_channels = in_channels
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        block = Bottleneck
        self.encoders = nn.ModuleList()
        for i in range(5):
            self.encoders.append(self._make_enc(block, planes[i], blocks[i], share_planes, stride=stride[i], nsample=nsample[i]))
        self.decoders = nn.ModuleList()
        for i in range(4, -1, -1):
            self.decoders.append(self._make_dec(block, planes[i], 2, share_planes, nsample=nsample[i], is_head=True if i == 4 else False))
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], num_classes))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """Private method to create encoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            stride: stride for pooling.
            nsample: number of neighbour to sample.

        Returns:
            Returns encoder object.
        """
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        """Private method to create decoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            nsample: number of neighbour to sample.
            is_head: bool type for head layer.

        Returns:
            Returns decoder object.
        """
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, batch):
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
                point (tf.float32): Input pointcloud (N,3)
                feat (tf.float32): Input features (N, 3)
                row_splits (tf.int64): row splits for batches (b+1,)

        Returns:
            Returns the probability distribution.
        """
        points = [batch.point]
        feats = [batch.feat]
        row_splits = [batch.row_splits]
        feats[0] = points[0] if self.in_channels == 3 else torch.cat((points[0], feats[0]), 1)
        for i in range(5):
            p, f, r = self.encoders[i]([points[i], feats[i], row_splits[i]])
            points.append(p)
            feats.append(f)
            row_splits.append(r)
        for i in range(4, -1, -1):
            if i == 4:
                feats[i + 1] = self.decoders[4 - i][1:]([points[i + 1], self.decoders[4 - i][0]([points[i + 1], feats[i + 1], row_splits[i + 1]]), row_splits[i + 1]])[1]
            else:
                feats[i + 1] = self.decoders[4 - i][1:]([points[i + 1], self.decoders[4 - i][0]([points[i + 1], feats[i + 1], row_splits[i + 1]], [points[i + 2], feats[i + 2], row_splits[i + 2]]), row_splits[i + 1]])[1]
        feat = self.cls(feats[1])
        return feat

    def preprocess(self, data, attr):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset. It consists of subsampling pointcloud with voxelization.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data

        """
        cfg = self.cfg
        points = np.array(data['point'], dtype=np.float32)
        if data.get('label') is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        if data.get('feat') is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)
        data = dict()
        if cfg.voxel_size:
            points_min = np.min(points, 0)
            points -= points_min
            if feat is None:
                sub_points, sub_labels = DataProcessing.grid_subsampling(points, labels=labels, grid_size=cfg.voxel_size)
                sub_feat = None
            else:
                sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(points, features=feat, labels=labels, grid_size=cfg.voxel_size)
        else:
            sub_points, sub_feat, sub_labels = points, feat, labels
        search_tree = KDTree(sub_points)
        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree
        if attr['split'] in ['test', 'testing']:
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds
        return data

    def transform(self, data, attr):
        """Transform function for the point cloud and features.

        This function is called after preprocess method. It consists
        of calling augmentation and normalizing the pointcloud.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns dictionary data with keys
            (point, feat, label).

        """
        cfg = self.cfg
        points = data['point']
        feat = data['feat']
        labels = data['label']
        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(points, feat, labels, self.cfg.get('augment', None))
        if attr['split'] not in ['test', 'testing']:
            if cfg.max_voxels and data['label'].shape[0] > cfg.max_voxels:
                init_idx = np.random.randint(labels.shape[0]) if 'train' in attr['split'] else labels.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(points - points[init_idx]), 1))[:cfg.max_voxels]
                if feat is not None:
                    points, feat, labels = points[crop_idx], feat[crop_idx], labels[crop_idx]
                else:
                    points, labels = points[crop_idx], labels[crop_idx]
        points_min, points_max = np.min(points, 0), np.max(points, 0)
        points -= (points_min + points_max) / 2.0
        data['point'] = torch.from_numpy(points)
        if feat is not None:
            data['feat'] = torch.from_numpy(feat) / 255.0
        data['label'] = torch.from_numpy(labels)
        return data

    def update_probs(self, inputs, results, test_probs, test_labels):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()
        labels = np.argmax(probs, 1)
        self.trans_point_sampler(patchwise=False)
        return probs, labels

    def inference_begin(self):
        data = self.preprocess(data, {'split': 'test'})
        data = self.transform(data, {'split': 'test'})
        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])
        reproj_inds = self.inference_input['proj_inds']
        probs = probs[reproj_inds]
        pred_l = np.argmax(probs, 1)
        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, sem_seg_loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Args:
            sem_seg_loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = inputs['data'].label
        scores, labels = filter_valid_label(results, labels, cfg.num_classes, cfg.ignored_label_inds, device)
        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)
        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.SGD(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(cfg_pipeline.max_epoch * 0.6), int(cfg_pipeline.max_epoch * 0.8)], gamma=0.1)
        return optimizer, scheduler


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name='', instance_norm=False, instance_norm_func=None):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv2d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)


class SharedMLP(nn.Sequential):

    def __init__(self, args: List[int], *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str='', instance_norm: bool=False):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact, instance_norm=instance_norm))


def _linear_bn_relu(in_channels, out_channels):
    """Layer combining Linear, BatchNorm and ReLU Block."""
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    """Creates multiple layered components. For each output channel,
    it creates Dense layers with Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        classifier: Whether the layer is classifier(appears at the end).
        dim: Dimension
        width_multiplier: factor by which neurons expands in intermediate layers.

    Returns:
        A List of layers.

    """
    r = width_multiplier
    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or len(out_channels) == 1 and out_channels[0] is None:
        return nn.Sequential(), in_channels, in_channels
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    elif classifier:
        layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
    else:
        layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


class SE3d(nn.Module):
    """Extra Sequential Dense layers to be used to increase
    model complexity.
    """

    def __init__(self, channel, reduction=8):
        """Constructor for SE3d module.

        Args:
            channel: Number of channels in the input layer.
            reduction: Factor of channels in second layer.

        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, inputs):
        """Forward call for SE3d

        Args:
            inputs: Input features.

        Returns:
            Transformed features.

        """
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)


def avg_voxelize(feat, coords, r):
    """Voxelize points and returns a voxel_grid with
    mean of features lying in same voxel.

    Args:
        feat: Input features (B, 3, N).
        coords: Input coordinates (B, C, N).
        r (int): Resolution of voxel grid.

    Returns:
        voxel grid (B, C, r, r, r)

    """
    coords = coords
    batch_size = feat.shape[0]
    dim = feat.shape[1]
    grid = torch.zeros((batch_size, dim, r, r, r))
    batch_id = torch.from_numpy(np.arange(batch_size).reshape(-1, 1))
    hash = batch_id * r * r * r + coords[:, 0, :] * r * r + coords[:, 1, :] * r + coords[:, 2, :]
    hash = hash.reshape(-1)
    for i in range(0, dim):
        grid_ = torch.zeros(batch_size * r * r * r, device=feat.device).scatter_add_(0, hash, feat[:, i, :].reshape(-1)).reshape(batch_size, r, r, r)
        grid[:, i] = grid_
    count = torch.zeros(batch_size * r * r * r, device=feat.device).scatter_add_(0, hash, torch.ones_like(feat[:, 0, :].reshape(-1))).reshape(batch_size, 1, r, r, r).clamp(min=1)
    count[count == 0] = 1
    grid = grid / count
    return grid


class Voxelization(nn.Module):
    """Voxelization module. Normalize the coordinates and
    returns voxel_grid with mean of features lying in same
    voxel.
    """

    def __init__(self, resolution, normalize=True, eps=1e-06):
        """Constructor of Voxelization module.

        Args:
            resolution (int): Resolution of the voxel grid.
            normalize (bool): Whether to normalize coordinates.
            eps (float): Small epsilon to avoid nan.

        """
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        """Forward pass for Voxelization.

        Args:
            features: Input features.
            coords: Input coordinates.

        Returns:
            Voxel grid of features (B, C, r, r, r)

        """
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords)
        return avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        """Extra representation of module."""
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class TrilinearDevoxelization(Function):

    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """Forward pass for the Op.

        Args:
            ctx: torch Autograd context.
            coords: the coordinates of points, FloatTensor[B, 3, N]
            features: FloatTensor[B, C, R, R, R]
            resolution: int, the voxel resolution.
            is_training: bool, training mode.
        
        Returns:
            torch.FloatTensor: devoxelized features (B, C, N)

        """
        B, C = features.shape[:2]
        features = features.contiguous()
        coords = coords.contiguous()
        outs, inds, wgts = trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for the Op.

        Args:
            ctx: torch Autograd context
            grad_output: gradient of outputs, FloatTensor[B, C, N]

        Returns:
            torch.FloatTensor: gradient of inputs (B, C, R, R, R)

        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply


class PVConv(nn.Module):
    """Point Voxel Convolution module. Consisting of 3D Convolutions
    for voxelized pointcloud, and SharedMLP blocks for point features.
    """

    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=1e-06):
        """Constructor for PVConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: kernel size for Conv3D.
            resolution: Resolution of the voxel grid.
            with_se: Whether to use extra dense layers in each block.
            normalize: Whether to normalize pointcloud before voxelization.
            eps: Epsilon for voxelization.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm3d(out_channels, eps=0.0001), nn.LeakyReLU(0.1, True), nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm3d(out_channels, eps=0.0001), nn.LeakyReLU(0.1, True)]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        """Forward pass for PVConv.

        Args:
            inputs: tuple of features and coordinates.

        Returns:
            Fused features consists of point features and
            voxel_features.

        """
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords


def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=1e-06, width_multiplier=1, voxel_resolution_multiplier=1):
    """Creates pointnet components. For each output channel,
    it comprises of PVConv or SharedMLP layers.

    Args:
        blocks: list of (out_channels, num_blocks, voxel_resolution).
        in_channels: Number of input channels.
        with_se: Whether to use extra dense layers in each block.
        normalize: Whether to normalize pointcloud before voxelization.
        eps: Epsilon for voxelization.
        width_multiplier: factor by which neurons expands in intermediate layers.
        voxel_resolution_multiplier: Factor by which voxel resolution expands.

    Returns:
        A List of layers, input_channels, and concat_channels

    """
    r, vr = width_multiplier, voxel_resolution_multiplier
    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), with_se=with_se, normalize=normalize, eps=eps)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels


class PVCNN(BaseModel):
    """Semantic Segmentation model. Based on Point Voxel Convolutions.
    https://arxiv.org/abs/1907.03739

    Uses PointNet architecture with separate Point and Voxel processing.

    Attributes:
        name: Name of model.
          Default to "PVCNN".
        num_classes: Number of classes.
        num_points: Number of points to sample per pointcloud.
        extra_feature_channels: Number of extra features.
          Default to 6 (RGB + Coordinate norms).
        batcher: Batching method for dataloader.
        augment: dictionary for augmentation.
    """
    blocks = (64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None)

    def __init__(self, name='PVCNN', device='cuda', num_classes=13, num_points=40960, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1, batcher='DefaultBatcher', augment=None, **kwargs):
        super(PVCNN, self).__init__(name=name, device=device, num_classes=num_classes, num_points=num_points, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier, batcher=batcher, augment=augment, **kwargs)
        cfg = self.cfg
        self.device = device
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.in_channels = extra_feature_channels + 3
        layers, channels_point, concat_channels_point = create_pointnet_components(blocks=self.blocks, in_channels=self.in_channels, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.point_features = nn.ModuleList(layers)
        layers, channels_cloud = create_mlp_components(in_channels=channels_point, out_channels=[256, 128], classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)
        layers, _ = create_mlp_components(in_channels=concat_channels_point + channels_cloud, out_channels=[512, 0.3, 256, 0.3, num_classes], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
                point (torch.float32): Input pointcloud (B, 3, N)
                feat (torch.float32): Input features (B, 9, N)

        Returns:
            torch.float32 : probability distribution (B, N, C).

        """
        coords = inputs['point']
        feat = inputs['feat']
        out_features_list = []
        for i in range(len(self.point_features)):
            feat, _ = self.point_features[i]((feat, coords))
            out_features_list.append(feat)
        feat = self.cloud_features(feat.max(dim=-1, keepdim=False).values)
        out_features_list.append(feat.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        out = self.classifier(torch.cat(out_features_list, dim=1))
        return out.transpose(1, 2)

    def preprocess(self, data, attr):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset. It consists of subsampling and normalizing the pointcloud and
        creating new features.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data

        """
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng
        points = np.array(data['point'], dtype=np.float32)
        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        if 'feat' not in data or data['feat'] is None:
            feat = points.copy()
        else:
            feat = np.array(data['feat'], dtype=np.float32)
        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(points, feat, labels, self.cfg.get('augment', None))
        points -= np.min(points, 0)
        feat = feat / 255.0
        max_points_x = np.max(points[:, 0])
        max_points_y = np.max(points[:, 1])
        max_points_z = np.max(points[:, 2])
        x, y, z = np.split(points, (1, 2), axis=-1)
        norm_x = x / max_points_x
        norm_y = y / max_points_y
        norm_z = z / max_points_z
        feat = np.concatenate([x, y, z, feat, norm_x, norm_y, norm_z], axis=-1)
        choices = rng.choice(points.shape[0], self.cfg.num_points, replace=points.shape[0] < self.cfg.num_points)
        points = points[choices].transpose()
        feat = feat[choices].transpose()
        labels = labels[choices]
        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels
        return data

    def transform(self, data, attr):
        """Transform function for the point cloud and features.

        This function is called after preprocess method. It consists
        of converting numpy arrays to torch Tensors.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns dictionary data with keys
            (point, feat, label).

        """
        data['point'] = torch.from_numpy(data['point'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])
        return data

    def update_probs(self, inputs, results, test_probs, test_labels):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()
        labels = np.argmax(probs, 1)
        self.trans_point_sampler(patchwise=False)
        return probs, labels

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})
        data['batch_lengths'] = [data['point'].shape[0]]
        data = self.transform(data, {})
        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])
        pred_l = np.argmax(probs, 1)
        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, sem_seg_loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Attributes:
            sem_seg_loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).
        
        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = inputs['data']['label'].reshape(-1)
        results = results.reshape(-1, results.shape[-1])
        scores, labels = filter_valid_label(results, labels, cfg.num_classes, cfg.ignored_label_inds, device)
        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)
        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler


default_collate_err_msg_format = 'default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}'


np_str_obj_array_pattern = re.compile('[SaUO]')


string_classes = str, bytes


def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))


class DefaultBatcher(object):
    """DefaultBatcher of PyTorch dataloader."""

    def __init__(self):
        super(DefaultBatcher, self).__init__()

    def collate_fn(self, batch):
        batching_result = default_collate(batch)
        return batching_result


class AttentivePooling(nn.Module):
    """This module pools down k neighbour features to a single encoding
    using weighted average with attention scores.
    """

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels, out_channels, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x):
        """Forward pass of the Module.

        Args:
            x: torch.Tensor of shape (B, dim_in, N, K).

        Returns:
            torch.Tensor of shape (B, d_out, N, 1).

        """
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)


class LocalSpatialEncoding(nn.Module):
    """This module computes k neighbour feature encoding for each point.
    Encoding consists of absolute distance, relative distance, positions.
    """

    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbor(self, coords, neighbor_indices):
        """Gather features based on neighbor indices.

        Args:
            coords: torch.Tensor of shape (B, N, d)
            neighbor_indices: torch.Tensor of shape (B, N, K)

        Returns:
            gathered neighbors of shape (B, dim, N, K)

        """
        B, N, K = neighbor_indices.size()
        dim = coords.shape[2]
        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, dim, N, K)
        neighbor_coords = torch.gather(extended_coords, 2, extended_indices)
        return neighbor_coords

    def forward(self, coords, features, neighbor_indices, relative_features=None):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                torch.Tensor of shape (B, N, 3)
            features: features of the pointcloud.
                torch.Tensor of shape (B, d, N, 1)
            neighbor_indices: indices of k neighbours.
                torch.Tensor of shape (B, N, K)
            relative_features: relative neighbor features calculated
              on first pass. Required for second pass.

        Returns:
            torch.Tensor of shape (B, 2*d, N, K)

        """
        B, N, K = neighbor_indices.size()
        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(torch.sum(torch.square(relative_pos), dim=1, keepdim=True))
            relative_features = torch.cat([relative_dist, relative_pos, extended_coords, neighbor_coords], dim=1)
        elif relative_features is None:
            raise ValueError('LocalSpatialEncoding: Require relative_features for second pass.')
        relative_features = self.mlp(relative_features)
        neighbor_features = self.gather_neighbor(features.transpose(1, 2).squeeze(3), neighbor_indices)
        return torch.cat([neighbor_features, relative_features], dim=1), relative_features


class LocalFeatureAggregation(nn.Module):
    """The neighbour features returned from LocalSpatialEncoding
    and pooled from AttentivePooling are aggregated and processed
    in multiple layers in this module.
    """

    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()
        self.num_neighbors = num_neighbors
        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, num_neighbors, encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                torch.Tensor of shape (B, N, 3).
            feat: features of the pointcloud.
                torch.Tensor of shape (B, d, N, 1)
            neighbor_indices: Indices of neighbors.

        Returns:
            torch.Tensor of shape (B, 2*d_out, N, 1).

        """
        x = self.mlp1(feat)
        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)
        x, _ = self.lse2(coords, x, neighbor_indices, relative_features=neighbor_features)
        x = self.pool2(x)
        return self.lrelu(self.mlp2(x) + self.shortcut(feat))


class RandLANet(BaseModel):
    """Class defining RandLANet, a Semantic Segmentation model.  Based on the
    architecture from the paper `RandLA-Net: Efficient Semantic Segmentation of
    Large-Scale Point Clouds <https://arxiv.org/abs/1911.11236>`__.

    RandLA-Net is an efficient and lightweight neural architecture which
    directly infer per-point semantics for large-scale point clouds. The key
    approach is to use random point sampling instead of more complex point
    selection approaches.  Although remarkably computation and memory
    efficient, random sampling can discard key features by chance. To overcome
    this, we introduce a novel local feature aggregation module to
    progressively increase the receptive field for each 3D point, thereby
    effectively preserving geometric details.

    **Architecture**

    .. image:: https://user-images.githubusercontent.com/23613902/150006228-34fb9e04-76b6-4022-af08-c308da6dcaae.png
        :width: 100%

    References:
        https://github.com/QingyongHu/RandLA-Net
    """

    def __init__(self, name='RandLANet', num_neighbors=16, num_layers=4, num_points=4096 * 11, num_classes=19, ignored_label_inds=[0], sub_sampling_ratio=[4, 4, 4, 4], in_channels=3, dim_features=8, dim_output=[16, 64, 128, 256], grid_size=0.06, batcher='DefaultBatcher', ckpt_path=None, augment={}, **kwargs):
        super().__init__(name=name, num_neighbors=num_neighbors, num_layers=num_layers, num_points=num_points, num_classes=num_classes, ignored_label_inds=ignored_label_inds, sub_sampling_ratio=sub_sampling_ratio, in_channels=in_channels, dim_features=dim_features, dim_output=dim_output, grid_size=grid_size, batcher=batcher, ckpt_path=ckpt_path, augment=augment, **kwargs)
        cfg = self.cfg
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)
        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-06, momentum=0.01)
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            self.encoder.append(LocalFeatureAggregation(dim_feature, cfg.dim_output[i], cfg.num_neighbors))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)
        self.encoder = nn.ModuleList(self.encoder)
        self.mlp = SharedMLP(dim_feature, dim_feature, activation_fn=nn.LeakyReLU(0.2))
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(SharedMLP(encoder_dim_list[-i - 2] + dim_feature, encoder_dim_list[-i - 2], transpose=True, activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]
        self.decoder = nn.ModuleList(self.decoder)
        self.fc1 = nn.Sequential(SharedMLP(dim_feature, 64, activation_fn=nn.LeakyReLU(0.2)), SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)), nn.Dropout(0.5), SharedMLP(32, cfg.num_classes, bn=False))

    def preprocess(self, data, attr):
        cfg = self.cfg
        points = np.array(data['point'][:, 0:3], dtype=np.float32)
        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        if 'feat' not in data or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)
        split = attr['split']
        data = dict()
        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(points, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(points, features=feat, labels=labels, grid_size=cfg.grid_size)
        search_tree = KDTree(sub_points)
        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree
        if split in ['test', 'testing']:
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds
        return data

    def transform(self, data, attr, min_possibility_idx=None):
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng
        cfg = self.cfg
        inputs = dict()
        pc = data['point'].copy()
        label = data['label'].copy()
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']
        pc, selected_idxs, center_point = self.trans_point_sampler(pc=pc, feat=feat, label=label, search_tree=tree, num_points=self.cfg.num_points)
        label = label[selected_idxs]
        if feat is not None:
            feat = feat[selected_idxs]
        augment_cfg = self.cfg.get('augment', {}).copy()
        val_augment_cfg = {}
        if 'recenter' in augment_cfg:
            val_augment_cfg['recenter'] = augment_cfg.pop('recenter')
        if 'normalize' in augment_cfg:
            val_augment_cfg['normalize'] = augment_cfg.pop('normalize')
        self.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)
        if attr['split'] in ['training', 'train']:
            pc, feat, label = self.augmenter.augment(pc, feat, label, augment_cfg, seed=rng)
        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)
        if cfg.in_channels != feat.shape[1]:
            raise RuntimeError('Wrong feature dimension, please update in_channels(3 + feature_dimension) in config')
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)
            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points
        inputs['coords'] = input_points
        inputs['neighbor_indices'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = feat
        inputs['point_inds'] = selected_idxs
        inputs['labels'] = label.astype(np.int64)
        return inputs

    def forward(self, inputs):
        """Forward pass for RandLANet

        Args:
            inputs: torch.Tensor, shape (B, N, d_in)
                input points

        Returns
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point

        """
        cfg = self.cfg
        feat = inputs['features']
        coords_list = [arr for arr in inputs['coords']]
        neighbor_indices_list = [arr for arr in inputs['neighbor_indices']]
        subsample_indices_list = [arr for arr in inputs['sub_idx']]
        interpolation_indices_list = [arr for arr in inputs['interp_idx']]
        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)
        l_relu = nn.LeakyReLU(0.2)
        feat = l_relu(feat)
        encoder_feat_list = []
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat, neighbor_indices_list[i])
            feat_sampled_i = self.random_sample(feat_encoder_i, subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(feat_encoder_i.clone())
            encoder_feat_list.append(feat_sampled_i.clone())
            feat = feat_sampled_i
        feat = self.mlp(feat)
        for i in range(cfg.num_layers):
            feat_interpolation_i = self.nearest_interpolation(feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = torch.cat([encoder_feat_list[-i - 2], feat_interpolation_i], dim=1)
            feat_decoder_i = self.decoder[i](feat_decoder_i)
            feat = feat_decoder_i
        scores = self.fc1(feat)
        return scores.squeeze(3).transpose(1, 2)

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        Args:
            feature: [B, d, N, 1] input features matrix
            pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling

        Returns:
             pool_features = [B, N', d] pooled features matrix

        """
        feature = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]
        pool_idx = torch.reshape(pool_idx, (batch_size, -1))
        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)
        feature = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features, (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        pool_features = pool_features.permute(0, 3, 1, 2)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        Args:
            feature: [B, d, N] input features matrix
            interp_idx: [B, up_num_points, 1] nearest neighbour index

        Returns:
             [B, up_num_points, d] interpolated features matrix

        """
        feature = feature.squeeze(3)
        d = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]
        interp_idx = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx = interp_idx.unsqueeze(1).expand(batch_size, d, -1)
        interpolatedim_features = torch.gather(feature, 2, interp_idx)
        interpolatedim_features = interpolatedim_features.unsqueeze(3)
        return interpolatedim_features

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Args:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model (B, N, C).
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.

        """
        cfg = self.cfg
        labels = inputs['data']['labels']
        scores, labels = filter_valid_label(results, labels, cfg.num_classes, cfg.ignored_label_inds, device)
        loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        return loss, labels, scores

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = self.rng.random(num_points) * 0.001
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes], dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        self.batcher = DefaultBatcher()

    def inference_preprocess(self):
        min_possibility_idx = np.argmin(self.possibility)
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, min_possibility_idx)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs
        return inputs

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])
        pred_l = np.argmax(probs, 1)
        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (1 - self.test_smooth) * probs
        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] - self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)
            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            inference_result = {'predict_labels': pred_labels, 'predict_scores': test_probs}
            data = self.inference_ori_data
            acc = (pred_labels == data['label'] - 1).mean()
            self.inference_result = inference_result
            return True
        else:
            return False

    def update_probs(self, inputs, results, test_probs, test_labels):
        """Update test probabilities with probs from current tested patch.

        Args:
            inputs: input to the model.
            results: output of the model.
            test_probs: probabilities for whole pointcloud
            test_labels: ground truth for whole pointcloud.

        Returns:
            updated probabilities and labels

        """
        self.test_smooth = 0.95
        for b in range(results.size()[0]):
            result = torch.reshape(results[b], (-1, self.cfg.num_classes))
            probs = torch.nn.functional.softmax(result, dim=-1)
            probs = probs.cpu().data.numpy()
            labels = np.argmax(probs, 1)
            inds = inputs['data']['point_inds'][b]
            test_probs[inds] = self.test_smooth * test_probs[inds] + (1 - self.test_smooth) * probs
            test_labels[inds] = labels
        return test_probs, test_labels


class InputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(InputLayer, self).__init__()
        self.voxel_size = torch.Tensor([voxel_size, voxel_size, voxel_size])

    def forward(self, features, in_positions):
        v = voxelize(in_positions, torch.LongTensor([0, in_positions.shape[0]]), self.voxel_size, torch.Tensor([0, 0, 0]), torch.Tensor([40960, 40960, 40960]))
        in_positions = in_positions[v.voxel_point_indices]
        features = features[v.voxel_point_indices]
        reverse_map_voxelize = np.zeros((in_positions.shape[0],))
        reverse_map_voxelize[v.voxel_point_indices.cpu().numpy()] = np.arange(in_positions.shape[0])
        reverse_map_voxelize = reverse_map_voxelize.astype(np.int32)
        in_positions = in_positions[v.voxel_point_row_splits[:-1]]
        count = v.voxel_point_row_splits[1:] - v.voxel_point_row_splits[:-1]
        reverse_map_sort = np.repeat(np.arange(count.shape[0]), count.cpu().numpy()).astype(np.int32)
        features_avg = in_positions.clone()
        features_avg[:, 0] = reduce_subarrays_sum(features[:, 0], v.voxel_point_row_splits)
        features_avg[:, 1] = reduce_subarrays_sum(features[:, 1], v.voxel_point_row_splits)
        features_avg[:, 2] = reduce_subarrays_sum(features[:, 2], v.voxel_point_row_splits)
        features_avg = features_avg / count.unsqueeze(1)
        return features_avg, in_positions, reverse_map_sort[reverse_map_voxelize]


class LinearBlock(nn.Module):

    def __init__(self, a, b):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(a, b)

    def forward(self, feat_list):
        out_list = []
        for feat in feat_list:
            out_list.append(self.linear(feat))
        return out_list

    def __name__(self):
        return 'LinearBlock'


class OutputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def forward(self, features_list, index_map_list):
        out = []
        for feat, index_map in zip(features_list, index_map_list):
            out.append(feat[index_map])
        return torch.cat(out, 0)


class ReLUBlock(nn.Module):

    def __init__(self):
        super(ReLUBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.relu(torch.cat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l
        return out_list

    def __name__(self):
        return 'ReLUBlock'


class SubmanifoldSparseConv(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, use_bias=False, offset=None, normalize=False):
        super(SubmanifoldSparseConv, self).__init__()
        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.0
            else:
                offset = 0.5
        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels, filters=filters, kernel_size=kernel_size, use_bias=use_bias, offset=offset, normalize=normalize)

    def forward(self, features_list, in_positions_list, out_positions_list=None, voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = in_positions_list
        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list, out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))
        return out_feat

    def __name__(self):
        return 'SubmanifoldSparseConv'


class ConcatFeat(nn.Module):

    def __init__(self):
        super(ConcatFeat, self).__init__()

    def __name__(self):
        return 'ConcatFeat'

    def forward(self, feat):
        return feat


def calculate_grid(in_positions):
    filter = torch.Tensor([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, -1, -1], [0, -1, 0], [0, 0, -1], [0, 0, 0]])
    out_pos = in_positions.long().repeat(1, filter.shape[0]).reshape(-1, 3)
    filter = filter.repeat(in_positions.shape[0], 1)
    out_pos = out_pos + filter
    out_pos = out_pos[out_pos.min(1).values >= 0]
    out_pos = out_pos[~(out_pos.long() % 2).bool().any(1)]
    out_pos = torch.unique(out_pos, dim=0)
    return out_pos + 0.5


class Convolution(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, use_bias=False, offset=None, normalize=False):
        super(Convolution, self).__init__()
        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.0
            else:
                offset = -0.5
        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels, filters=filters, kernel_size=kernel_size, use_bias=use_bias, offset=offset, normalize=normalize)

    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_positions_list = []
        for in_positions in in_positions_list:
            out_positions_list.append(calculate_grid(in_positions))
        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list, out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))
        out_positions_list = [(out / 2) for out in out_positions_list]
        return out_feat, out_positions_list

    def __name__(self):
        return 'Convolution'


class DeConvolution(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, use_bias=False, offset=None, normalize=False):
        super(DeConvolution, self).__init__()
        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.0
            else:
                offset = -0.5
        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConvTranspose(in_channels=in_channels, filters=filters, kernel_size=kernel_size, use_bias=use_bias, offset=offset, normalize=normalize)

    def forward(self, features_list, in_positions_list, out_positions_list, voxel_size=1.0):
        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list, out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))
        return out_feat

    def __name__(self):
        return 'DeConvolution'


class JoinFeat(nn.Module):

    def __init__(self):
        super(JoinFeat, self).__init__()

    def __name__(self):
        return 'JoinFeat'

    def forward(self, feat_cat, feat):
        out = []
        for a, b in zip(feat_cat, feat):
            out.append(torch.cat([a, b], -1))
        return out


class NetworkInNetwork(nn.Module):

    def __init__(self, nIn, nOut, bias=False):
        super(NetworkInNetwork, self).__init__()
        if nIn == nOut:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(nIn, nOut, bias=bias)

    def forward(self, inputs):
        out = []
        for inp in inputs:
            out.append(self.linear(inp))
        return out


class ResidualBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super(ResidualBlock, self).__init__()
        self.lin = NetworkInNetwork(nIn, nOut)
        self.batch_norm1 = BatchNormBlock(nIn)
        self.relu1 = ReLUBlock()
        self.sub_sparse_conv1 = SubmanifoldSparseConv(in_channels=nIn, filters=nOut, kernel_size=[3, 3, 3])
        self.batch_norm2 = BatchNormBlock(nOut)
        self.relu2 = ReLUBlock()
        self.sub_sparse_conv2 = SubmanifoldSparseConv(in_channels=nOut, filters=nOut, kernel_size=[3, 3, 3])

    def forward(self, feat_list, pos_list):
        out1 = self.lin(feat_list)
        feat_list = self.batch_norm1(feat_list)
        feat_list = self.relu1(feat_list)
        feat_list = self.sub_sparse_conv1(feat_list, pos_list)
        feat_list = self.batch_norm2(feat_list)
        feat_list = self.relu2(feat_list)
        out2 = self.sub_sparse_conv2(feat_list, pos_list)
        return [(a + b) for a, b in zip(out1, out2)]

    def __name__(self):
        return 'ResidualBlock'


class UNet(nn.Module):

    def __init__(self, conv_block_reps, nPlanes, residual_blocks=False, downsample=[2, 2], leakiness=0):
        super(UNet, self).__init__()
        self.net = nn.ModuleList(self.get_UNet(nPlanes, residual_blocks, conv_block_reps))
        self.residual_blocks = residual_blocks

    @staticmethod
    def block(layers, a, b, residual_blocks):
        if residual_blocks:
            layers.append(ResidualBlock(a, b))
        else:
            layers.append(BatchNormBlock(a))
            layers.append(ReLUBlock())
            layers.append(SubmanifoldSparseConv(in_channels=a, filters=b, kernel_size=[3, 3, 3]))

    @staticmethod
    def get_UNet(nPlanes, residual_blocks, conv_block_reps):
        layers = []
        for i in range(conv_block_reps):
            UNet.block(layers, nPlanes[0], nPlanes[0], residual_blocks)
        if len(nPlanes) > 1:
            layers.append(ConcatFeat())
            layers.append(BatchNormBlock(nPlanes[0]))
            layers.append(ReLUBlock())
            layers.append(Convolution(in_channels=nPlanes[0], filters=nPlanes[1], kernel_size=[2, 2, 2]))
            layers = layers + UNet.get_UNet(nPlanes[1:], residual_blocks, conv_block_reps)
            layers.append(BatchNormBlock(nPlanes[1]))
            layers.append(ReLUBlock())
            layers.append(DeConvolution(in_channels=nPlanes[1], filters=nPlanes[0], kernel_size=[2, 2, 2]))
            layers.append(JoinFeat())
            for i in range(conv_block_reps):
                UNet.block(layers, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0], residual_blocks)
        return layers

    def forward(self, pos_list, feat_list):
        conv_pos = []
        concat_feat = []
        for module in self.net:
            if isinstance(module, BatchNormBlock):
                feat_list = module(feat_list)
            elif isinstance(module, ReLUBlock):
                feat_list = module(feat_list)
            elif isinstance(module, ResidualBlock):
                feat_list = module(feat_list, pos_list)
            elif isinstance(module, SubmanifoldSparseConv):
                feat_list = module(feat_list, pos_list)
            elif isinstance(module, Convolution):
                conv_pos.append([pos.clone() for pos in pos_list])
                feat_list, pos_list = module(feat_list, pos_list)
            elif isinstance(module, DeConvolution):
                feat_list = module(feat_list, [(2 * pos) for pos in pos_list], conv_pos[-1])
                pos_list = conv_pos.pop()
            elif isinstance(module, ConcatFeat):
                concat_feat.append([feat.clone() for feat in module(feat_list)])
            elif isinstance(module, JoinFeat):
                feat_list = module(concat_feat.pop(), feat_list)
            else:
                raise Exception('Unknown module {}'.format(module))
        return feat_list


class SparseConvUnet(BaseModel):
    """Semantic Segmentation model.

    Uses UNet architecture replacing convolutions with Sparse Convolutions.

    Attributes:
        name: Name of model.
          Default to "SparseConvUnet".
        device: Which device to use (cpu or cuda).
        voxel_size: Voxel length for subsampling.
        multiplier: min length of feature length in each layer.
        conv_block_reps: repetition of Unet Blocks.
        residual_blocks: Whether to use Residual Blocks.
        in_channels: Number of features(default 3 for color).
        num_classes: Number of classes.
    """

    def __init__(self, name='SparseConvUnet', device='cuda', multiplier=16, voxel_size=0.05, conv_block_reps=1, residual_blocks=False, in_channels=3, num_classes=20, grid_size=4096, batcher='ConcatBatcher', augment=None, **kwargs):
        super(SparseConvUnet, self).__init__(name=name, device=device, multiplier=multiplier, voxel_size=voxel_size, conv_block_reps=conv_block_reps, residual_blocks=residual_blocks, in_channels=in_channels, num_classes=num_classes, grid_size=grid_size, batcher=batcher, augment=augment, **kwargs)
        cfg = self.cfg
        self.device = device
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)
        self.multiplier = cfg.multiplier
        self.input_layer = InputLayer()
        self.sub_sparse_conv = SubmanifoldSparseConv(in_channels=in_channels, filters=multiplier, kernel_size=[3, 3, 3])
        self.unet = UNet(conv_block_reps, [multiplier, 2 * multiplier, 3 * multiplier, 4 * multiplier, 5 * multiplier, 6 * multiplier, 7 * multiplier], residual_blocks)
        self.batch_norm = BatchNormBlock(multiplier)
        self.relu = ReLUBlock()
        self.linear = LinearBlock(multiplier, num_classes)
        self.output_layer = OutputLayer()

    def forward(self, inputs):
        pos_list = []
        feat_list = []
        index_map_list = []
        for i in range(len(inputs.batch_lengths)):
            pos = inputs.point[i]
            feat = inputs.feat[i]
            feat, pos, index_map = self.input_layer(feat, pos)
            pos_list.append(pos)
            feat_list.append(feat)
            index_map_list.append(index_map)
        feat_list = self.sub_sparse_conv(feat_list, pos_list, voxel_size=1.0)
        feat_list = self.unet(pos_list, feat_list)
        feat_list = self.batch_norm(feat_list)
        feat_list = self.relu(feat_list)
        feat_list = self.linear(feat_list)
        output = self.output_layer(feat_list, index_map_list)
        return output

    def preprocess(self, data, attr):
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(torch.utils.data.get_worker_info().seed + torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng
        points = np.array(data['point'], dtype=np.float32)
        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        if 'feat' not in data or data['feat'] is None:
            raise Exception("SparseConvnet doesn't work without feature values.")
        feat = np.array(data['feat'], dtype=np.float32)
        points *= 1.0 / self.cfg.voxel_size
        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(points, feat, labels, self.cfg.get('augment', None), seed=rng)
        m = points.min(0)
        M = points.max(0)
        grid_size = self.cfg.grid_size
        offset = -m + np.clip(grid_size - M + m - 0.001, 0, None) * rng.random(3) + np.clip(grid_size - M + m + 0.001, None, 0) * rng.random(3)
        points += offset
        idxs = (points.min(1) >= 0) * (points.max(1) < 4096)
        points = points[idxs]
        feat = feat[idxs]
        labels = labels[idxs]
        points = (points.astype(np.int32) + 0.5).astype(np.float32)
        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels
        return data

    def transform(self, data, attr):
        data['point'] = torch.from_numpy(data['point'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])
        return data

    def update_probs(self, inputs, results, test_probs, test_labels):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()
        labels = np.argmax(probs, 1)
        self.trans_point_sampler(patchwise=False)
        return probs, labels

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})
        data['batch_lengths'] = [data['point'].shape[0]]
        data = self.transform(data, {})
        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])
        pred_l = np.argmax(probs, 1)
        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Attributes:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).
        
        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = torch.cat(inputs['data'].label, 0)
        scores, labels = filter_valid_label(results, labels, cfg.num_classes, cfg.ignored_label_inds, device)
        loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) ->torch.Tensor:
        """Forward.

        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        idx = ball_query(xyz, new_xyz, radius, nsample)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query_gpu = BallQuery.apply


class QueryAndGroup(nn.Module):

    def __init__(self, radius: float, nsample: int, use_xyz: bool=True):
        """QueryAndGroup.

        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) ->Tuple[torch.Tensor]:
        """Forward.

        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError
        batch_size = xyz.shape[0]
        idx = ball_query_gpu(self.radius, self.nsample, xyz, new_xyz)
        idx_stacked = torch.stack([idx] * 3, dim=1).view(batch_size, 3, -1).long()
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = torch.gather(xyz_trans, dim=2, index=idx_stacked).view(batch_size, 3, -1, self.nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            idx_stacked = torch.stack([idx] * features.shape[1], dim=1).view(batch_size, features.shape[1], -1).long()
            grouped_features = torch.gather(features, dim=2, index=idx_stacked).view(batch_size, features.shape[1], -1, self.nsample)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features


class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        """Forward.

        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Anchor3DHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatFeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (JoinFeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBlock,
     lambda: ([], {'a': 4, 'b': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetworkInNetwork,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutputLayer,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (PFNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SECOND,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     False),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransitionDown,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
]

class Test_isl_org_Open3D_ML(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

