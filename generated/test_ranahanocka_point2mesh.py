import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
layers = _module
mesh = _module
mesh_conv = _module
mesh_pool = _module
mesh_union = _module
mesh_unpool = _module
losses = _module
networks = _module
options = _module
blender_hull = _module
convex_hull = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import time


import copy


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import ConstantPad2d


from typing import Union


from torch.nn import init


from torch import optim


from typing import List


import uuid


class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """

    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        G = self.create_GeMM(x, G)
        x = self.conv(G)
        return x

    def flatten_gemm_inds(self, Gi):
        b, ne, nn = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])
        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)
        x_1 = f[:, :, :, (1)] + f[:, :, :, (3)]
        x_2 = f[:, :, :, (2)] + f[:, :, :, (4)]
        x_3 = torch.abs(f[:, :, :, (1)] - f[:, :, :, (3)])
        x_4 = torch.abs(f[:, :, :, (2)] - f[:, :, :, (4)])
        f = torch.stack([f[:, :, :, (0)], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), 'constant', 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm


class MeshUnion:

    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        self.groups[(target), :] += self.groups[(source), :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[(edge_key), :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[(tensor_mask), :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        fe = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe

    def prepare_groups(self, features, mask):
        tensor_mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[(tensor_mask), :], 0, 1).transpose_(1, 0)
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)


class MeshPool(nn.Module):

    def __init__(self, target):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        self.__fe = fe
        self.__meshes = meshes
        for mesh_index in range(len(meshes)):
            self.__pool_main(mesh_index)
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        fe = self.__fe[(mesh_index), :, :mesh.edges_count]
        in_fe_sq = torch.sum(fe ** 2, dim=0)
        sorted, edge_ids = torch.sort(in_fe_sq, descending=True)
        edge_ids = edge_ids.tolist()
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        while mesh.edges_count > self.__out_target:
            edge_id = edge_ids.pop()
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0) and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) and self.__is_one_ring_valid(mesh, edge_id):
            self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(edge_groups, key_b, key_a)
        MeshPool.__union_groups(edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert len(shared_items) == 2
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(edge_groups, key_a, edge_id)
            MeshPool.__union_groups(edge_groups, key_b, edge_id)
            MeshPool.__union_groups(edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - side_a % 2 + 2) % 4
        other_side_b = (side_b - side_b % 2 + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert len(vertex) == 1
        mesh.remove_vertex(vertex[0])

    @staticmethod
    def __union_groups(edge_groups, source, target):
        edge_groups.union(source, target)

    @staticmethod
    def __remove_group(edge_groups, index):
        edge_groups.remove_group(index)


class MeshUnpool(nn.Module):

    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows = unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols != 0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1)
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        occurrences = occurrences.expand(unroll_mat.shape)
        unroll_mat = unroll_mat / occurrences
        unroll_mat = unroll_mat
        for mesh in meshes:
            mesh.unroll_gemm()
        return torch.matmul(features, unroll_mat)


class ConvBlock(nn.Module):

    def __init__(self, in_feat, out_feat, k=1):
        super(ConvBlock, self).__init__()
        self.lst = [MeshConv(in_feat, out_feat)]
        for i in range(k - 1):
            self.lst.append(MeshConv(out_feat, out_feat))
        self.lst = nn.ModuleList(self.lst)

    def forward(self, input, meshes):
        for c in self.lst:
            input = c(input, meshes)
        return input


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True, batch_norm=True, transfer_data=True, leaky=0):
        super(UpConv, self).__init__()
        self.leaky = leaky
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = ConvBlock(in_channels, out_channels)
        if transfer_data:
            self.conv1 = ConvBlock(2 * out_channels, out_channels)
        else:
            self.conv1 = ConvBlock(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def forward(self, x, from_down=None):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        if self.bn:
            x1 = self.bn[0](x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


class MeshDecoder(nn.Module):

    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, leaky=0):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll, batch_norm=batch_norm, transfer_data=transfer_data, leaky=leaky))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False, batch_norm=batch_norm, transfer_data=False, leaky=leaky)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=0, pool=0, leaky=0):
        super(DownConv, self).__init__()
        self.leaky = leaky
        self.bn = []
        self.pool = None
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def forward(self, x):
        fe, meshes = x[0], x[1]
        x1 = self.conv1(fe, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        if self.bn:
            x1 = self.bn[0](x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.leaky_relu(x2, self.leaky)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class MeshEncoder(nn.Module):

    def __init__(self, pools, convs, blocks=0, leaky=0):
        super(MeshEncoder, self).__init__()
        self.leaky = leaky
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool, leaky=leaky))
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        return fe, encoder_outs


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True, leaky=0):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks, leaky=leaky)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data, leaky=leaky)
        self.bn = nn.InstanceNorm2d(up_convs[-1])

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        fe = self.bn(fe.unsqueeze(-1))
        return fe, None


def build_v(x, meshes):
    mesh = meshes[0]
    x = x.reshape(len(meshes), 2, 3, -1)
    vs_to_sum = torch.zeros([len(meshes), len(mesh.vs_in), mesh.max_nvs, 3], dtype=x.dtype, device=x.device)
    x = x[:, (mesh.vei), :, (mesh.ve_in)].transpose(0, 1)
    vs_to_sum[:, (mesh.nvsi), (mesh.nvsin), :] = x
    vs_sum = torch.sum(vs_to_sum, dim=2)
    nvs = mesh.nvs
    vs = vs_sum / nvs[(None), :, (None)]
    return vs


def init_weights(net, init_type, init_gain):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class PriorNet(nn.Module):
    """
    network for
    """

    def __init__(self, n_edges, in_ch=6, convs=[32, 64], pool=[], res_blocks=0, init_verts=None, transfer_data=False, leaky=0, init_weights_size=0.002):
        super(PriorNet, self).__init__()
        down_convs = [in_ch] + convs
        up_convs = convs[::-1] + [in_ch]
        pool_res = [n_edges] + pool
        self.encoder_decoder = MeshEncoderDecoder(pools=pool_res, down_convs=down_convs, up_convs=up_convs, blocks=res_blocks, transfer_data=transfer_data, leaky=leaky)
        self.last_conv = MeshConv(6, 6)
        init_weights(self, 'normal', init_weights_size)
        eps = 1e-08
        self.last_conv.conv.weight.data.uniform_(-1 * eps, eps)
        self.last_conv.conv.bias.data.uniform_(-1 * eps, eps)
        self.init_verts = init_verts

    def forward(self, x, meshes):
        meshes_new = [i.deep_copy() for i in meshes]
        x, _ = self.encoder_decoder(x, meshes_new)
        x = x.squeeze(-1)
        x = self.last_conv(x, meshes_new).squeeze(-1)
        est_verts = build_v(x.unsqueeze(0), meshes)
        return est_verts.float() + self.init_verts.expand_as(est_verts)


class PartNet(PriorNet):

    def __init__(self, init_part_mesh, in_ch=6, convs=[32, 64], pool=[], res_blocks=0, init_verts=None, transfer_data=False, leaky=0, init_weights_size=0.002):
        temp = torch.linspace(len(convs), 1, len(convs)).long().tolist()
        super().__init__(temp[0], in_ch=in_ch, convs=convs, pool=temp[1:], res_blocks=res_blocks, init_verts=init_verts, transfer_data=transfer_data, leaky=leaky, init_weights_size=init_weights_size)
        self.mesh_pools = []
        self.mesh_unpools = []
        self.factor_pools = pool
        for i in self.modules():
            if isinstance(i, MeshPool):
                self.mesh_pools.append(i)
            if isinstance(i, MeshUnpool):
                self.mesh_unpools.append(i)
        self.mesh_pools = sorted(self.mesh_pools, key=lambda x: x._MeshPool__out_target, reverse=True)
        self.mesh_unpools = sorted(self.mesh_unpools, key=lambda x: x.unroll_target, reverse=False)
        self.init_part_verts = nn.ParameterList([torch.nn.Parameter(i) for i in init_part_mesh.init_verts])
        for i in self.init_part_verts:
            i.requires_grad = False

    def __set_pools(self, n_edges: int, new_pools: List[int]):
        for i, l in enumerate(self.mesh_pools):
            l._MeshPool__out_target = new_pools[i]
        new_pools = [n_edges] + new_pools
        new_pools = new_pools[:-1]
        new_pools.reverse()
        for i, l in enumerate(self.mesh_unpools):
            l.unroll_target = new_pools[i]

    def forward(self, x, partmesh):
        """
        forward PartNet
        :param x: BXfXn_edges
        :param partmesh:
        :return:
        """
        for i, p in enumerate(partmesh):
            n_edges = p.edges_count
            self.init_verts = self.init_part_verts[i]
            temp_pools = [int(n_edges - i) for i in self.make3(PartNet.array_times(n_edges, self.factor_pools))]
            self.__set_pools(n_edges, temp_pools)
            relevant_edges = x[:, :, (partmesh.sub_mesh_edge_index[i])]
            results = super().forward(relevant_edges, [p])
            yield results

    @staticmethod
    def array_times(num: int, iterable):
        return [(i * num) for i in iterable]

    @staticmethod
    def make3(array):
        diff = [(i % 3) for i in array]
        return [(array[i] - diff[i]) for i in range(len(array))]

