import sys
_module = sys.modules[__name__]
del sys
auxiliary = _module
argument_parser = _module
html_report = _module
init_html_report = _module
meter = _module
my_utils = _module
Shuffle = _module
parallel_shuffle = _module
shuffle = _module
sampling_and_meshing = _module
visualization = _module
augmenter = _module
dataset_shapenet = _module
mesh_processor = _module
pointcloud_processor = _module
trainer_dataset = _module
atlasnet = _module
model = _module
model_blocks = _module
resnet = _module
template = _module
trainer_model = _module
train = _module
launcher = _module
metro = _module
trainer = _module
trainer_abstract = _module
trainer_iteration = _module
trainer_loss = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import random


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


from copy import deepcopy


import torch.nn as nn


import torch.nn.parallel


import torch.utils.data


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


import torch.optim as optim


import time


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def get_activation(argument):
    getter = {'relu': F.relu, 'sigmoid': F.sigmoid, 'softplus': F.softplus, 'logsigmoid': F.logsigmoid, 'softsign': F.softsign, 'tanh': F.tanh}
    return getter.get(argument, 'Invalid activation')


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        self.opt = opt
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers
        super(Mapping2Dto3D, self).__init__()
        None
        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList([torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])
        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)
        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])
        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.opt.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)


class Template(object):

    def get_random_points(self):
        None

    def get_regular_points(self):
        None


class SphereTemplate(Template):

    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    def get_random_points(self, shape, device='gpu0'):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, 'shape should have 3 in dim 1'
        rand_grid = torch.cuda.FloatTensor(shape).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None, device='gpu0'):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.mesh = pymesh.generate_icosphere(1, [0, 0, 0], 4)
            self.vertex = torch.from_numpy(self.mesh.vertices).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0, 1).contiguous().unsqueeze(0)
            self.npoints = npoints
        return Variable(self.vertex)


class SquareTemplate(Template):

    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device='gpu0'):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device='gpu0'):
        """
        Get regular points on a Square
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)
            self.vertex = torch.from_numpy(self.mesh.vertices).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0, 1).contiguous().unsqueeze(0)
        return Variable(self.vertex[:, :2].contiguous())

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])
        for i in range(1, int(grain + 1)):
            for j in range(0, int(grain + 1) - 1):
                faces.append([j + (grain + 1) * i, j + (grain + 1) * i + 1, j + (grain + 1) * (i - 1)])
        for i in range(0, int(grain + 1) - 1):
            for j in range(1, int(grain + 1)):
                faces.append([j + (grain + 1) * i, j + (grain + 1) * i - 1, j + (grain + 1) * (i + 1)])
        return np.array(vertices), np.array(faces)


def get_template(template_type, device=0):
    getter = {'SQUARE': SquareTemplate, 'SPHERE': SphereTemplate}
    template = getter.get(template_type, 'Invalid template')
    return template(device=device)


class Atlasnet(nn.Module):

    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(Atlasnet, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.nb_pts_in_primitive = opt.number_points // opt.nb_primitives
        self.nb_pts_in_primitive_eval = opt.number_points_eval // opt.nb_primitives
        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            None
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        if train:
            input_points = [self.template[i].get_random_points(torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)), latent_vector.device) for i in range(self.opt.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval, device=latent_vector.device) for i in range(self.opt.nb_primitives)]
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).unsqueeze(1) for i in range(0, self.opt.nb_primitives)], dim=1)
        return output_points.contiguous()

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0) == 1, 'input should have batch size 1!'
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device) for i in range(self.opt.nb_primitives)]
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in range(0, self.opt.nb_primitives)]
        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(), faces=self.template[i].mesh.faces) for i in range(self.opt.nb_primitives)]
        mesh = pymesh.merge_meshes(output_meshes)
        return mesh


class PointNet(nn.Module):

    def __init__(self, nlatent=1024, dim_input=3):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """
        super(PointNet, self).__init__()
        self.dim_input = dim_input
        self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)
        self.nlatent = nlatent

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        if opt.SVR:
            self.encoder = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
        else:
            self.encoder = PointNet(nlatent=opt.bottleneck_size)
        self.decoder = Atlasnet(opt)
        self
        if not opt.SVR:
            self.apply(weights_init)
        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PointNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
]

class Test_ThibaultGROUEIX_AtlasNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

