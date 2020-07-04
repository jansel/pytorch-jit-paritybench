import sys
_module = sys.modules[__name__]
del sys
SMIL_torch_batch = _module
preprocess = _module
smpl_np = _module
smpl_tf = _module
smpl_torch = _module
smpl_torch_batch = _module
test = _module

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


import torch.nn as nn


import numpy as np


import scipy.sparse


from torch.nn import Module


from time import time


class SMIL(nn.Module):

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a batch of [3, 4] matrices.

        Parameter:
        ---------
        x: Tensor to be appended of shape [N, 3, 4]

        Return:
        ------
        Tensor after appending of shape [N, 4, 4]

        """
        ret = torch.cat([x, self.e4.expand(x.shape[0], 1, -1)], dim=1)
        return ret

    def pack(self, x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensors.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        ret = torch.cat((torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=
            x.dtype, device=x.device), x), dim=3)
        return ret

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [N, 1, 3].

        Return:
        -------
        Rotation matrix of shape [N, 3, 3].
        """
        theta = torch.norm(r, dim=(1, 2), keepdim=True)
        torch.max(theta, theta.new_full((1,), torch.finfo(theta.dtype).tiny
            ), out=theta)
        r_hat = r / theta
        z_stick = torch.zeros_like(r_hat[:, (0), (0)])
        m = torch.stack((z_stick, -r_hat[:, (0), (2)], r_hat[:, (0), (1)],
            r_hat[:, (0), (2)], z_stick, -r_hat[:, (0), (0)], -r_hat[:, (0),
            (1)], r_hat[:, (0), (0)], z_stick), dim=1)
        m = m.reshape(-1, 3, 3)
        dot = torch.bmm(r_hat.transpose(1, 2), r_hat)
        cos = theta.cos()
        R = cos * self.eye + (1 - cos) * dot + theta.sin() * m
        return R

    def __init__(self, model_path='./model.pkl', sparse=True):
        super().__init__()
        self.parent = None
        self.model_path = None
        if model_path is not None:
            with open(model_path, 'rb') as f:
                self.model_path = model_path
                params = pickle.load(f)
                registerbuffer = lambda name: self.register_buffer(name,
                    torch.as_tensor(params[name]))
                registerbuffer('weights')
                registerbuffer('posedirs')
                registerbuffer('v_template')
                registerbuffer('shapedirs')
                self.register_buffer('f', torch.as_tensor(params['f'].
                    astype(np.int32)))
                self.register_buffer('kintree_table', torch.as_tensor(
                    params['kintree_table'].astype(np.int32)))
                J_regressor = params['J_regressor']
                if scipy.sparse.issparse(J_regressor):
                    J_regressor = J_regressor.tocoo()
                    J_regressor = torch.sparse_coo_tensor([J_regressor.row,
                        J_regressor.col], J_regressor.data, J_regressor.shape)
                    if not sparse:
                        J_regressor = J_regressor.to_dense()
                else:
                    J_regressor = torch.as_tensor(J_regressor)
                self.register_buffer('J_regressor', J_regressor)
                self.register_buffer('e4', self.posedirs.new_tensor([0, 0, 
                    0, 1]))
                self.register_buffer('eye', torch.eye(3, dtype=self.e4.
                    dtype, device=self.e4.device))
                self.set_parent()
        self._register_state_dict_hook(self.set_parent)

    def set_parent(self, *args, **kwargs):
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self
            .kintree_table.shape[1])}
        self.parent = {i: id_to_col[self.kintree_table[0, i].item()] for i in
            range(1, self.kintree_table.shape[1])}

    def save_obj(self, verts, obj_mesh_name):
        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.f:
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def regress_joints(self, vertices):
        """The J_regressor matrix transforms vertices to joints."""
        batch_size = vertices.shape[0]
        batch_vertices = vertices.transpose(0, 1).reshape(self.J_regressor.
            shape[1], -1)
        batch_results = self.J_regressor.mm(batch_vertices)
        batch_results = batch_results.reshape(self.J_regressor.shape[0],
            batch_size, -1).transpose(0, 1)
        return batch_results

    def rotate_translate(self, rotation_matrix, translation):
        transform = torch.cat((rotation_matrix, translation.unsqueeze(2)), 2)
        return self.with_zeros(transform)

    def forward(self, beta, pose, trans=None, simplify=False):
        """This module takes betas and poses in a batched manner.
        A pose is 3 * K + 3 (= self.kintree_table.shape[1] * 3) parameters, where K is the number of joints.
        A beta is a vector of size self.shapedirs.shape[2], that parameterizes the body shape.
        Since this is batched, multiple betas and poses should be concatenated along zeroth dimension.
        See http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf for more info.
        """
        batch_size = beta.shape[0]
        v_shaped = torch.tensordot(beta, self.shapedirs, dims=([1], [2])
            ) + self.v_template
        R_cube = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_size,
            -1, 3, 3)
        J = self.regress_joints(v_shaped)
        if not simplify:
            lrotmin = R_cube[:, 1:] - self.eye
            lrotmin = lrotmin.reshape(batch_size, -1)
            v_shaped += torch.tensordot(lrotmin, self.posedirs, dims=([1], [2])
                )
        rest_shape_h = torch.cat((v_shaped, v_shaped.new_ones(1).expand(*
            v_shaped.shape[:-1], 1)), 2)
        G = [self.rotate_translate(R_cube[:, (0)], J[:, (0)])]
        for i in range(1, self.kintree_table.shape[1]):
            G.append(torch.bmm(G[self.parent[i]], self.rotate_translate(
                R_cube[:, (i)], J[:, (i)] - J[:, (self.parent[i])])))
        G = torch.stack(G, 1)
        Jtr = G[(...), :4, (3)].clone()
        G = G - self.pack(torch.matmul(G, torch.cat([J, J.new_zeros(1).
            expand(*J.shape[:2], 1)], dim=2).unsqueeze(-1)))
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3,
            1, 2)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_size, -1, 4, 1))
            ).reshape(batch_size, -1, 4)
        if trans is not None:
            trans = trans.unsqueeze(1)
            v[(...), :3] += trans
            Jtr[(...), :3] += trans
        return v, Jtr


class SMPLModel(Module):

    def __init__(self, device=None, model_path='./model.pkl'):
        super(SMPLModel, self).__init__()
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].
            todense())).type(torch.float64)
        self.weights = torch.from_numpy(params['weights']).type(torch.float64)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64
            )
        self.v_template = torch.from_numpy(params['v_template']).type(torch
            .float64)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.
            float64)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        for name in ['J_regressor', 'weights', 'posedirs', 'v_template',
            'shapedirs']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor)

    @staticmethod
    def rodrigues(r):
        """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
        eps = r.clone().normal_(std=1e-08)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float64)
        m = torch.stack((z_stick, -r_hat[:, (0), (2)], r_hat[:, (0), (1)],
            r_hat[:, (0), (2)], z_stick, -r_hat[:, (0), (0)], -r_hat[:, (0),
            (1)], r_hat[:, (0), (0)], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = torch.eye(3, dtype=torch.float64).unsqueeze(dim=0
            ) + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        ret = torch.cat((x, ones), dim=0)
        return ret

    @staticmethod
    def pack(x):
        """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
        zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float64)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in (self.faces + 1):
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans, simplify=False):
        """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.
            kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1,
            self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])
            ) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))
        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = torch.eye(3, dtype=torch.float64).unsqueeze(dim=0
                ) + torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin,
                dims=([2], [0]))
        results = []
        results.append(self.with_zeros(torch.cat((R_cube_big[0], torch.
            reshape(J[(0), :], (3, 1))), dim=1)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(torch.matmul(results[parent[i]], self.with_zeros
                (torch.cat((R_cube_big[i], torch.reshape(J[(i), :] - J[(
                parent[i]), :], (3, 1))), dim=1))))
        stacked = torch.stack(results, dim=0)
        results = stacked - self.pack(torch.matmul(stacked, torch.reshape(
            torch.cat((J, torch.zeros((24, 1), dtype=torch.float64)), dim=1
            ), (24, 4, 1))))
        T = torch.tensordot(self.weights, results, dims=([1], [0]))
        rest_shape_h = torch.cat((v_posed, torch.ones((v_posed.shape[0], 1),
            dtype=torch.float64)), dim=1)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        result = v + torch.reshape(trans, (1, 3))
        return result


class SMPLModel(Module):

    def __init__(self, device=None, model_path='./model.pkl'):
        super(SMPLModel, self).__init__()
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].
            todense())).type(torch.float64)
        if 'joint_regressor' in params.keys():
            self.joint_regressor = torch.from_numpy(np.array(params[
                'joint_regressor'].T.todense())).type(torch.float64)
        else:
            self.joint_regressor = torch.from_numpy(np.array(params[
                'J_regressor'].todense())).type(torch.float64)
        self.weights = torch.from_numpy(params['weights']).type(torch.float64)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64
            )
        self.v_template = torch.from_numpy(params['v_template']).type(torch
            .float64)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.
            float64)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        for name in ['J_regressor', 'joint_regressor', 'weights',
            'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            None
            setattr(self, name, _tensor)

    @staticmethod
    def rodrigues(r):
        """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
        eps = r.clone().normal_(std=1e-08)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float64)
        m = torch.stack((z_stick, -r_hat[:, (0), (2)], r_hat[:, (0), (1)],
            r_hat[:, (0), (2)], z_stick, -r_hat[:, (0), (0)], -r_hat[:, (0),
            (1)], r_hat[:, (0), (0)], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = torch.eye(3, dtype=torch.float64).unsqueeze(dim=0
            ) + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
        ones = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
            ).expand(x.shape[0], -1, -1)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
        zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.
            float64)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in (self.faces + 1):
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans, simplify=False):
        """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.
          
          20190128: Add batch support.

          Parameters:
          ---------
          pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [N, 3].

          Return:
          ------
          A 3-D tensor of [N * 6890 * 3] for vertices,
          and the corresponding [N * 19 * 3] joint positions.

    """
        batch_num = betas.shape[0]
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.
            kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1,
            self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])
            ) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3)).reshape(batch_num,
            -1, 3, 3)
        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = torch.eye(3, dtype=torch.float64).unsqueeze(dim=0
                ) + torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=
                torch.float64)
            lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2
                )
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs,
                dims=([1], [2]))
        results = []
        results.append(self.with_zeros(torch.cat((R_cube_big[:, (0)], torch
            .reshape(J[:, (0), :], (-1, 3, 1))), dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(torch.matmul(results[parent[i]], self.with_zeros
                (torch.cat((R_cube_big[:, (i)], torch.reshape(J[:, (i), :] -
                J[:, (parent[i]), :], (-1, 3, 1))), dim=2))))
        stacked = torch.stack(results, dim=1)
        results = stacked - self.pack(torch.matmul(stacked, torch.reshape(
            torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.
            float64)), dim=2), (batch_num, 24, 4, 1))))
        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(
            0, 3, 1, 2)
        rest_shape_h = torch.cat((v_posed, torch.ones((batch_num, v_posed.
            shape[1], 1), dtype=torch.float64)), dim=2)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
        result = v + torch.reshape(trans, (batch_num, 1, 3))
        joints = torch.tensordot(result, self.joint_regressor, dims=([1], [0])
            ).transpose(1, 2)
        return result, joints


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_CalciferZh_SMPL(_paritybench_base):
    pass
