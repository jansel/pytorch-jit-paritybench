import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
net = _module
train_dcgan = _module
updater = _module
visualize = _module
writetensorboard = _module
data = _module
train_vae = _module
create_wit_samples = _module
demo = _module
demo_beholder = _module
demo_caffe2 = _module
demo_custom_scalars = _module
demo_embedding = _module
demo_global_writer = _module
demo_graph = _module
demo_hogwild = _module
demo_hparams = _module
demo_matplotlib = _module
demo_multiple_embedding = _module
demo_multiprocessing = _module
demo_nvidia_smi = _module
demo_onnx = _module
demo_openvino = _module
demo_purge = _module
global_1 = _module
global_2 = _module
setup = _module
tensorboardX = _module
beholder = _module
file_system_tools = _module
shared_config = _module
video_writing = _module
caffe2_graph = _module
crc32c = _module
embedding = _module
event_file_writer = _module
global_writer = _module
onnx_graph = _module
openvino_graph = _module
proto = _module
api_pb2 = _module
attr_value_pb2 = _module
event_pb2 = _module
graph_pb2 = _module
layout_pb2 = _module
node_def_pb2 = _module
plugin_hparams_pb2 = _module
plugin_mesh_pb2 = _module
plugin_pr_curve_pb2 = _module
plugin_text_pb2 = _module
resource_handle_pb2 = _module
step_stats_pb2 = _module
summary_pb2 = _module
tensor_pb2 = _module
tensor_shape_pb2 = _module
types_pb2 = _module
versions_pb2 = _module
proto_graph = _module
pytorch_graph = _module
record_writer = _module
summary = _module
torchvis = _module
utils = _module
visdom_writer = _module
writer = _module
x2num = _module
tests = _module
event_file_writer_test = _module
expect_reader = _module
record_writer_test = _module
test_beholder = _module
test_caffe2 = _module
test_chainer_np = _module
test_crc32c = _module
test_embedding = _module
test_figure = _module
test_hparams = _module
test_lint = _module
test_numpy = _module
test_onnx_graph = _module
test_openvino_graph = _module
test_pr_curve = _module
test_pytorch_graph = _module
test_pytorch_np = _module
test_record_writer = _module
test_summary = _module
test_summary_writer = _module
test_utils = _module
test_visdom = _module
test_writer = _module
tset_multiprocess_write = _module

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


import torch.nn.functional as F


from torch.autograd.variable import Variable


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.autograd import Variable


import torch.optim as optim


import torch.multiprocessing as mp


import logging


from collections import OrderedDict


class M(nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.cn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=128, out_features=2)

    def forward(self, i):
        i = self.cn1(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = self.cn2(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = i.view(len(i), -1)
        i = self.fc1(i)
        i = F.log_softmax(i, dim=1)
        return i


class LinearInLinear(nn.Module):

    def __init__(self):
        super(LinearInLinear, self).__init__()
        self.l = nn.Linear(3, 5)

    def forward(self, x):
        return self.l(x)


class MultipleInput(nn.Module):

    def __init__(self):
        super(MultipleInput, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)

    def forward(self, x, y):
        return self.Linear_1(x + y)


class MultipleOutput(nn.Module):

    def __init__(self):
        super(MultipleOutput, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)
        self.Linear_2 = nn.Linear(3, 7)

    def forward(self, x):
        return self.Linear_1(x), self.Linear_2(x)


class MultipleOutput_shared(nn.Module):

    def __init__(self):
        super(MultipleOutput_shared, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)

    def forward(self, x):
        return self.Linear_1(x), self.Linear_1(x)


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = Net1()

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


n_categories = 10


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
            hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size,
            output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden, input

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lanpa_tensorboardX(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(LinearInLinear(*[], **{}), [torch.rand([3, 3])], {})

    def test_002(self):
        self._check(MultipleOutput(*[], **{}), [torch.rand([3, 3])], {})

    def test_003(self):
        self._check(MultipleOutput_shared(*[], **{}), [torch.rand([3, 3])], {})

    def test_004(self):
        self._check(SimpleModel(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

