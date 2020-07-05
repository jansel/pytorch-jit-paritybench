import sys
_module = sys.modules[__name__]
del sys
auxiliary = _module
argument_parser = _module
dataset = _module
meter = _module
model = _module
my_utils = _module
ply = _module
pointcloud_processor = _module
visualization = _module
data = _module
generate_data_animals = _module
generate_data_humans = _module
correspondences = _module
launch = _module
script = _module
test_zip = _module
test_zip_2 = _module
abstract_trainer = _module
train = _module
trainer = _module
LaplacianLoss = _module
laplacian = _module
train_unsup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
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


import torch.nn.parallel


import torch.utils.data


from torch.autograd import Variable


import numpy as np


import torch.nn.functional as F


import torch.optim as optim


import time


from scipy import sparse


class PointNetfeat(nn.Module):

    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)
        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class patchDeformationMLP(nn.Module):
    """ Deformation of a 2D patch into a 3D surface """

    def __init__(self, patchDim=2, patchDeformDim=3, tanh=True):
        super(patchDeformationMLP, self).__init__()
        layer_size = 128
        self.tanh = tanh
        self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x


class PointGenCon(nn.Module):

    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        None
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x


class GetTemplate(object):

    def __init__(self, start_from, dataset_train=None):
        if start_from == 'TEMPLATE':
            self.init_template()
        elif start_from == 'SOUP':
            self.init_soup()
        elif start_from == 'TRAININGDATA':
            self.init_trainingdata(dataset_train)
        else:
            print('select valid template type')

    def init_template(self):
        if not os.path.exists('./data/template/template.ply'):
            os.system('chmod +x ./data/download_template.sh')
            os.system('./data/download_template.sh')
        mesh = trimesh.load('./data/template/template.ply', process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)
        mesh_HR = trimesh.load('./data/template/template_dense.ply', process=False)
        self.mesh_HR = mesh_HR
        point_set_HR = mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)
        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert np.abs(np.sum(self.prop) - 1) < 0.001, 'Propabilities do not sum to 1!)'
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()
        print(f'Using template to initialize template')

    def init_soup(self):
        mesh = trimesh.load('./data/template/template.ply', process=False)
        self.mesh = mesh
        self.vertex = torch.FloatTensor(6890, 3).normal_().cuda()
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f'Using Random soup to initialize template')

    def init_trainingdata(self, dataset_train=None):
        mesh = trimesh.load('./data/template/template.ply', process=False)
        self.mesh = mesh
        index = np.random.randint(len(dataset_train))
        points = dataset_train.datas[index].squeeze().clone()
        self.vertex = points
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f'Using training data number {index} to initialize template')


class AE_AtlasNet_Humans(nn.Module):

    def __init__(self, num_points=6890, bottleneck_size=1024, point_translation=False, dim_template=3, patch_deformation=False, dim_out_patch=3, start_from='TEMPLATE', dataset_train=None):
        super(AE_AtlasNet_Humans, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.point_translation = point_translation
        self.dim_template = dim_template
        self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0
        self.start_from = start_from
        self.dataset_train = dataset_train
        self.template = [GetTemplate(start_from, dataset_train)]
        self.dim_before_decoder = dim_template
        if self.dim_template > 3:
            self.template[0].vertex = torch.cat([self.template[0].vertex, torch.zeros((self.template[0].vertex.size(0), self.dim_template - 3))], -1)
            self.template[0].vertex_HR = torch.cat([self.template[0].vertex_HR, torch.zeros((self.template[0].vertex_HR.size(0), self.dim_template - 3))], -1)
        elif self.dim_template == 2:
            self.template[0].vertex = self.template[0].vertex[:, :2]
            self.template[0].vertex_HR = self.template[0].vertex_HR[:, :2]
        if patch_deformation:
            self.dim_before_decoder = dim_out_patch
            self.templateDiscovery = nn.ModuleList([patchDeformationMLP(patchDim=dim_template, patchDeformDim=dim_out_patch, tanh=True)])
        if point_translation:
            self.template[0].vertex_trans = torch.nn.Parameter(self.template[0].vertex.clone().zero_())
            self.register_parameter('template_vertex_' + str(0), self.template[0].vertex_trans)
        self.encoder = PointNetfeat(num_points, bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size)])

    def morph_points(self, x, idx=None):
        if not idx is None:
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
        rand_grid = self.template[0].vertex
        if not idx is None:
            rand_grid = rand_grid[(idx), :]
            rand_grid = rand_grid.view(x.size(0), -1, self.dim_template).transpose(1, 2).contiguous()
        else:
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(x.size(0), self.dim_template, -1)
        if self.patch_deformation:
            rand_grid = self.templateDiscovery[0](rand_grid)
        if self.point_translation:
            if idx is None:
                trans = self.template[0].vertex_trans.unsqueeze(0).transpose(1, 2).contiguous().expand(x.size(0), self.dim_template, -1)
            else:
                trans = self.template[0].vertex_trans[(idx), :].view(x.size(0), -1, self.dim_template).transpose(1, 2).contiguous()
            rand_grid = rand_grid + trans
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        return self.decoder[0](y).contiguous().transpose(2, 1).contiguous()

    def decode(self, x, idx=None):
        return self.morph_points(x, idx)

    def forward(self, x, idx=None):
        x = self.encoder(x)
        return self.decode(x, idx)

    def decode_full(self, x):
        outs = []
        div = 20
        batch = int(self.template[0].num_vertex_HR / div)
        for i in range(div - 1):
            rand_grid = self.template[0].template_learned_HR[batch * i:batch * (i + 1)].view(x.size(0), batch, self.dim_template).transpose(1, 2).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.synchronize()
        i = div - 1
        rand_grid = self.template[0].template_learned_HR[batch * i:].view(x.size(0), -1, self.dim_template).transpose(1, 2).contiguous()
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.synchronize()
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def get_points_translation_template(self):
        base_shape = self.template[0].vertex
        if self.patch_deformation:
            base_shape = self.get_patch_deformation_template()[0]
        return [base_shape + self.template[0].vertex_trans]

    def get_patch_deformation_template(self, high_res=False):
        self.eval()
        None
        if high_res:
            rand_grid = self.template[0].vertex_HR.transpose(0, 1).contiguous().unsqueeze(0).expand(1, self.dim_template, -1)
        else:
            rand_grid = self.template[0].vertex.transpose(0, 1).contiguous().unsqueeze(0).expand(1, self.dim_template, -1)
        return [self.templateDiscovery[0](rand_grid).squeeze().transpose(1, 0).contiguous()]

    def make_high_res_template_from_low_res(self):
        """
        This function takes a path to the orginal shapenet model and subsample it nicely
        """
        if not (self.point_translation or self.patch_deformation):
            self.template[0].template_learned_HR = self.template[0].vertex_HR
        if self.patch_deformation:
            templates = self.get_patch_deformation_template(high_res=True)
            self.template[0].template_learned_HR = templates[0]
        if self.point_translation:
            templates = self.get_points_translation_template()
            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                obj1 = pymesh.form_mesh(vertices=template_points, faces=self.template[0].mesh.faces)
                if len(obj1.vertices) < 100000:
                    obj1 = pymesh.split_long_edges(obj1, 0.02)[0]
                    while len(obj1.vertices) < 100000:
                        obj1 = pymesh.subdivide(obj1)
                self.template[0].mesh_HR = obj1
                self.template[0].template_learned_HR = torch.from_numpy(obj1.vertices).float()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                None
            elif self.dim_template == 2:
                templates = templates[0]
                templates = torch.cat([templates, torch.zeros((templates.size(0), 1))], -1)
                template_points = templates.cpu().clone().detach().numpy()
                obj1 = pymesh.form_mesh(vertices=template_points, faces=self.template[0].mesh.faces)
                if len(obj1.vertices) < 100000:
                    obj1 = pymesh.split_long_edges(obj1, 0.02)[0]
                    while len(obj1.vertices) < 100000:
                        obj1 = pymesh.subdivide(obj1)
                self.template[0].mesh_HR = obj1
                self.template[0].template_learned_HR = torch.from_numpy(obj1.vertices).float()[:, :2].contiguous()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                None
            else:
                template_points = templates[0].cpu().clone().detach().numpy()
                self.template[0].mesh_HR = self.template[0].mesh
                self.template[0].template_learned_HR = torch.from_numpy(template_points).float()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                None

    def save_template_png(self, path):
        None
        self.eval()
        if self.point_translation:
            templates = self.get_points_translation_template()
            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                mesh_point_translation = trimesh.Trimesh(vertices=template_points, faces=self.template[0].mesh.faces, process=False)
                mesh_point_translation.export(os.path.join(path, 'mesh_point_translation.ply'))
                p1 = template_points[:, (0)]
                p2 = template_points[:, (1)]
                p3 = template_points[:, (2)]
                fig = plt.figure(figsize=(20, 20), dpi=80)
                fig.set_size_inches(20, 20)
                ax = fig.add_subplot(111, projection='3d', facecolor='white')
                ax.view_init(0, 30)
                ax.set_xlim3d(-0.8, 0.8)
                ax.set_ylim3d(-0.8, 0.8)
                ax.set_zlim3d(-0.8, 0.8)
                ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                plt.grid(b=None)
                plt.axis('off')
                fig.savefig(os.path.join(path, 'points_' + str(0) + '_' + str(self.count)), bbox_inches='tight', pad_inches=0)
            else:
                None
        if self.patch_deformation:
            templates = self.get_patch_deformation_template()
            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                mesh_patch_deformation = trimesh.Trimesh(vertices=template_points, faces=self.template[0].mesh.faces, process=False)
                mesh_patch_deformation.export(os.path.join(path, 'mesh_patch_deformation.ply'))
                p1 = template_points[:, (0)]
                p2 = template_points[:, (1)]
                p3 = template_points[:, (2)]
                fig = plt.figure(figsize=(20, 20), dpi=80)
                fig.set_size_inches(20, 20)
                ax = fig.add_subplot(111, projection='3d', facecolor='white')
                ax.view_init(0, 30)
                ax.set_xlim3d(-0.8, 0.8)
                ax.set_ylim3d(-0.8, 0.8)
                ax.set_zlim3d(-0.8, 0.8)
                ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                plt.grid(b=None)
                plt.axis('off')
                fig.savefig(os.path.join(path, 'deformation_' + str(0) + '_' + str(self.count)), bbox_inches='tight', pad_inches=0)
            else:
                None
        self.count += 1


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PointGenCon,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2500, 64])], {}),
     True),
    (PointNetfeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
    (patchDeformationMLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64])], {}),
     True),
]

class Test_ThibaultGROUEIX_3D_CODED(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

