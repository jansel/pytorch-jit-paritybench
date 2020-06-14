import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
batch_svd = _module
create_UV_maps = _module
data_washing = _module
densebody_dataset = _module
preprocess_smpl = _module
procrustes = _module
smpl_torch_batch = _module
triangulation = _module
uv_map_generator = _module
uv_map_generator_unit_test = _module
visualizer = _module
eval = _module
models = _module
base_model = _module
create_model = _module
networks = _module
resnet_model = _module
nohup_train = _module
test = _module
train = _module

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


import numpy as np


import torch


from torch.nn import Module


from time import time


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from numpy.linalg import solve


from scipy.interpolate import RectBivariateSpline as RBS


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


class SMPLModel(Module):

    def __init__(self, device=None, model_path='./model.pkl', data_type=
        torch.float, simplify=False):
        super(SMPLModel, self).__init__()
        self.data_type = data_type
        self.simplify = simplify
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].
            todense())).type(self.data_type)
        self.joint_regressor = torch.from_numpy(np.array(params[
            'joint_regressor'].T.todense())).type(self.data_type)
        self.weights = torch.from_numpy(params['weights']).type(self.data_type)
        self.posedirs = torch.from_numpy(params['posedirs']).type(self.
            data_type)
        self.v_template = torch.from_numpy(params['v_template']).type(self.
            data_type)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(self.
            data_type)
        self.kintree_table = params['kintree_table']
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.
            kintree_table.shape[1])}
        self.parent = {i: id_to_col[self.kintree_table[0, i]] for i in
            range(1, self.kintree_table.shape[1])}
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        self.visualize_model_parameters()
        for name in ['J_regressor', 'joint_regressor', 'weights',
            'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            None
            setattr(self, name, _tensor.to(device))

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
        z_stick = torch.zeros(theta_dim, dtype=r.dtype).to(r.device)
        m = torch.stack((z_stick, -r_hat[:, (0), (2)], r_hat[:, (0), (1)],
            r_hat[:, (0), (2)], z_stick, -r_hat[:, (0), (0)], -r_hat[:, (0),
            (1)], r_hat[:, (0), (0)], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=r.dtype).unsqueeze(dim=0) + torch.
            zeros((theta_dim, 3, 3), dtype=r.dtype)).to(r.device)
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
        ones = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=x.dtype).expand(x
            .shape[0], -1, -1).to(x.device)
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
        zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=x.dtype
            ).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in (self.faces + 1):
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def visualize_model_parameters(self):
        self.write_obj(self.v_template, 'v_template.obj')
    """
    _lR2G: Buildin function, calculating G terms for each vertex.
  """

    def _lR2G(self, lRs, J):
        batch_num = lRs.shape[0]
        results = []
        results.append(self.with_zeros(torch.cat((lRs[:, (0)], torch.
            reshape(J[:, (0), :], (-1, 3, 1))), dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(torch.matmul(results[self.parent[i]], self.
                with_zeros(torch.cat((lRs[:, (i)], torch.reshape(J[:, (i),
                :] - J[:, (self.parent[i]), :], (-1, 3, 1))), dim=2))))
        stacked = torch.stack(results, dim=1)
        deformed_joint = torch.matmul(stacked, torch.reshape(torch.cat((J,
            torch.zeros((batch_num, 24, 1), dtype=self.data_type).to(self.
            device)), dim=2), (batch_num, 24, 4, 1)))
        results = stacked - self.pack(deformed_joint)
        return results, lRs

    def theta2G(self, thetas, J):
        batch_num = thetas.shape[0]
        lRs = self.rodrigues(thetas.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3
            )
        return self._lR2G(lRs, J)
    """
    gR2G: Calculate G terms from global rotation matrices.
    --------------------------------------------------
    Input: gR: global rotation matrices [N * 24 * 3 * 3]
           J: shape blended template pose J(b)
  """

    def gR2G(self, gR, J):
        lRs = [gR[:, (0)]]
        for i in range(1, self.kintree_table.shape[1]):
            lRs.append(torch.bmm(gR[:, (self.parent[i])].transpose(1, 2),
                gR[:, (i)]))
        lRs = torch.stack(lRs, dim=1)
        return self._lR2G(lRs, J)

    def forward(self, betas, thetas, trans, gR=None):
        """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.
          
          20190128: Add batch support.
          20190322: Extending forward compatiability with SMPLModelv3
          
          Usage:
          ---------
          meshes, joints = forward(betas, thetas, trans): normal SMPL 
          meshes, joints = forward(betas, thetas, trans, gR=gR): 
                calling from SMPLModelv3, using gR to cache G terms, ignoring thetas

          Parameters:
          ---------
          thetas: an [N, 24 * 3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [N, 3].
          
          G, R_cube_big: (Added on 0322) Fix compatible issue when calling from v3 objects
            when calling this mode, theta must be set as None
          
          Return:
          ------
          A 3-D tensor of [N * 6890 * 3] for vertices,
          and the corresponding [N * 24 * 3] joint positions.

    """
        batch_num = betas.shape[0]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])
            ) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        if gR is not None:
            G, R_cube_big = self.gR2G(gR, J)
        elif thetas is not None:
            G, R_cube_big = self.theta2G(thetas, J)
        else:
            raise RuntimeError(
                'Either thetas or gR should be specified, but detected two Nonetypes'
                )
        if self.simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=self.data_type).unsqueeze(dim=0) +
                torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=self.
                data_type)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(batch_num, -1)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs,
                dims=([1], [2]))
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3,
            1, 2)
        rest_shape_h = torch.cat((v_posed, torch.ones((batch_num, v_posed.
            shape[1], 1), dtype=self.data_type).to(self.device)), dim=2)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
        result = v + torch.reshape(trans, (batch_num, 1, 3))
        joints = torch.tensordot(result, self.joint_regressor, dims=([1], [0])
            ).transpose(1, 2)
        return result, joints


def acquire_weights(UV_weight_npy):
    if os.path.isfile(UV_weight_npy):
        return np.load(UV_weight_npy)
    else:
        mask_name = UV_weight_npy.replace('weights.npy', 'mask.png')
        print(mask_name)
        UV_mask = imread(mask_name)
        if UV_mask.ndim == 3:
            UV_mask = UV_mask[:, :, (0)]
        ret, labels = connectedComponents(UV_mask, connectivity=4)
        unique, counts = np.unique(labels, return_counts=True)
        print(unique, counts)
        UV_weights = np.zeros_like(UV_mask).astype(np.float32)
        for id, count in zip(unique, counts):
            if id == 0:
                continue
            indices = np.argwhere(labels == id)
            UV_weights[indices[:, (0)], indices[:, (1)]] = 1 / count
        UV_weights *= np.prod(UV_mask.shape)
        np.save(UV_weight_npy, UV_weights)
        return UV_weights


class WeightedL1Loss(nn.Module):

    def __init__(self, uv_map, device):
        super(WeightedL1Loss, self).__init__()
        self.weight = torch.from_numpy(acquire_weights(
            'data_utils/{}_UV_weights.npy'.format(uv_map))).to(device)
        None
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input * self.weight, target * self.weight)


class TotalVariationLoss(nn.Module):

    def __init__(self, uv_map, device):
        super(TotalVariationLoss, self).__init__()
        weight = torch.from_numpy(acquire_weights(
            'data_utils/{}_UV_weights.npy'.format(uv_map))).to(device)
        self.weight = weight[0:-1, 0:-1]
        self.factor = self.weight.shape[0] * self.weight.shape[1]

    def __call__(self, input):
        tv = torch.abs(input[:, :, 0:-1, 0:-1] - input[:, :, 0:-1, 1:]
            ) + torch.abs(input[:, :, 0:-1, 0:-1] - input[:, :, 1:, 0:-1])
        return torch.sum(tv * self.weight) / self.factor


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicResBlock(nn.Module):

    def __init__(self, inplanes, norm_layer=nn.BatchNorm2d,
        activation_layer=nn.LeakyReLU(0.2, True)):
        super(BasicResBlock, self).__init__()
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.inplanes = inplanes
        layers = [conv3x3(inplanes, inplanes), norm_layer(inplanes),
            activation_layer, conv3x3(inplanes, inplanes), norm_layer(inplanes)
            ]
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return self.res(x) + x


def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size
        =3, stride=1, padding=0))


class ConvResBlock(nn.Module):

    def __init__(self, inplanes, planes, direction, stride=1, norm_layer=nn
        .BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(ConvResBlock, self).__init__()
        self.res = BasicResBlock(inplanes, norm_layer=norm_layer,
            activation_layer=activation_layer)
        self.activation = activation_layer
        if stride == 1 and inplanes == planes:
            conv = lambda x: x
        elif direction == 'down':
            conv = conv3x3(inplanes, planes, stride=stride)
        elif direction == 'up':
            conv = deconv3x3(inplanes, planes, stride=stride)
        else:
            raise ValueError(
                'Direction must be either "down" or "up", get %s instead.' %
                direction)
        self.conv = conv
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.conv(self.activation(self.res(x)))


class ResNetEncoder(nn.Module):

    def __init__(self, im_size, nz=256, ngf=64, ndown=6, norm_layer=None,
        nl_layer=None):
        super(ResNetEncoder, self).__init__()
        self.ngf = ngf
        fc_dim = 2 * nz
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7,
            stride=1, padding=0), norm_layer(ngf), nl_layer]
        prev = 1
        for i in range(ndown):
            im_size //= 2
            cur = min(8, prev * 2)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction=
                'down', stride=2, norm_layer=norm_layer, activation_layer=
                nl_layer))
            prev = cur
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(im_size * im_size * ngf * cur,
            fc_dim), nn.BatchNorm1d(fc_dim), nl_layer, nn.Linear(fc_dim, nz))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class VGGEncoder(nn.Module):

    def __init__(self, im_size, nz=256, ngf=64, ndown=5, norm_layer=None,
        nl_layer=None):
        super(VGGEncoder, self).__init__()
        cfg_parts = [[1 * ngf, 1 * ngf, 'M'], [2 * ngf, 2 * ngf, 'M'], [4 *
            ngf, 4 * ngf, 'M'], [8 * ngf, 8 * ngf, 'M']]
        custom_cfg = []
        for i in range(ndown):
            custom_cfg += cfg_parts[min(i, 3)]
        fc_dim = 4 * nz
        self.features = self._make_layers(cfg=custom_cfg, batch_norm=True,
            norm_layer=norm_layer, nl_layer=nl_layer)
        im_size = im_size // 2 ** ndown
        self.avgpool = nn.AdaptiveAvgPool2d((im_size, im_size))
        self.classifier = nn.Sequential(nn.Linear(512 * im_size * im_size,
            fc_dim), nl_layer, nn.Dropout(), nn.Linear(fc_dim, nz))

    def _make_layers(self, cfg, batch_norm=False, norm_layer=None, nl_layer
        =None):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), nl_layer]
                else:
                    layers += [conv2d, nl_layer]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ConvResDecoder(nn.Module):
    """
        ConvResDecoder: Use convres block for upsampling
    """

    def __init__(self, im_size, nz, ngf=64, nup=6, norm_layer=None,
        nl_layer=None):
        super(ConvResDecoder, self).__init__()
        self.im_size = im_size // 2 ** nup
        fc_dim = 2 * nz
        layers = []
        prev = 8
        for i in range(nup - 1, -1, -1):
            cur = min(prev, 2 ** i)
            layers.append(ConvResBlock(ngf * prev, ngf * cur, direction=
                'up', stride=2, norm_layer=norm_layer, activation_layer=
                nl_layer))
            prev = cur
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7,
            stride=1, padding=0), nn.Tanh()]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(nz, fc_dim), nn.BatchNorm1d(
            fc_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(fc_dim, 
            self.im_size * self.im_size * ngf * 8))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)


class ConvUpSampleDecoder(nn.Module):
    """
        SimpleDecoder
    """

    def __init__(self, im_size, nz, ngf=64, nup=6, norm_layer=None,
        nl_layer=None):
        super(ConvUpSampleDecoder, self).__init__()
        self.im_size = im_size // 2 ** nup
        fc_dim = 4 * nz
        layers = []
        prev = 8
        for i in range(nup - 1, -1, -1):
            cur = min(prev, 2 ** i)
            layers.append(deconv3x3(ngf * prev, ngf * cur, stride=2))
            prev = cur
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7,
            stride=1, padding=0), nn.Tanh()]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(nz, fc_dim), nl_layer, nn.Dropout
            (), nn.Linear(fc_dim, self.im_size * self.im_size * ngf * 8))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.im_size, self.im_size)
        return self.conv(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Lotayou_densebody_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicResBlock(*[], **{'inplanes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ConvResBlock(*[], **{'inplanes': 4, 'planes': 4, 'direction': 4}), [torch.rand([4, 4, 4, 4])], {})

