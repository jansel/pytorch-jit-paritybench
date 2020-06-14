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
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.
            bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.
            bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.
            bottleneck_size // 4, 1)
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
        mesh_HR = trimesh.load('./data/template/template_dense.ply',
            process=False)
        self.mesh_HR = mesh_HR
        point_set_HR = mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(
            point_set_HR)
        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert np.abs(np.sum(self.prop) - 1
            ) < 0.001, 'Propabilities do not sum to 1!)'
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ThibaultGROUEIX_3D_CODED(_paritybench_base):
    pass
    def test_000(self):
        self._check(PointGenCon(*[], **{}), [torch.rand([4, 2500, 64])], {})

    def test_001(self):
        self._check(PointNetfeat(*[], **{}), [torch.rand([4, 3, 64])], {})

    def test_002(self):
        self._check(patchDeformationMLP(*[], **{}), [torch.rand([4, 2, 64])], {})

