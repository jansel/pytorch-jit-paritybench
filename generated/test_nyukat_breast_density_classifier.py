import sys
_module = sys.modules[__name__]
del sys
convert_model = _module
density_model = _module
density_model_tf = _module
density_model_torch = _module
layers = _module
layers_tf = _module
layers_torch = _module
models = _module
models_tf = _module
models_torch = _module
test_inference = _module
utils = _module

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


import collections as col


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std, device):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.device = device

    def forward(self, x):
        if not self.gaussian_noise_std:
            return x
        return {'L-CC': self._add_gaussian_noise(x['L-CC']), 'L-MLO': self.
            _add_gaussian_noise(x['L-MLO']), 'R-CC': self.
            _add_gaussian_noise(x['R-CC']), 'R-MLO': self.
            _add_gaussian_noise(x['R-MLO'])}

    def _add_gaussian_noise(self, single_view):
        return single_view + torch.Tensor(*single_view.shape).normal_(std=
            self.gaussian_noise_std).to(self.device)


class AllViewsConvLayer(nn.Module):
    """Convolutional layers across all 4 views"""

    def __init__(self, in_channels, number_of_filters=32, filter_size=(3, 3
        ), stride=(1, 1)):
        super(AllViewsConvLayer, self).__init__()
        self.cc = nn.Conv2d(in_channels=in_channels, out_channels=
            number_of_filters, kernel_size=filter_size, stride=stride)
        self.mlo = nn.Conv2d(in_channels=in_channels, out_channels=
            number_of_filters, kernel_size=filter_size, stride=stride)

    def forward(self, x):
        return {'L-CC': F.relu(self.cc(x['L-CC'])), 'L-MLO': F.relu(self.
            mlo(x['L-MLO'])), 'R-CC': F.relu(self.cc(x['R-CC'])), 'R-MLO':
            F.relu(self.mlo(x['R-MLO']))}

    @property
    def ops(self):
        return {'CC': self.cc, 'MLO': self.mlo}


class AllViewsMaxPool(nn.Module):
    """Max-pool across all 4 views"""

    def __init__(self):
        super(AllViewsMaxPool, self).__init__()

    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return {'L-CC': F.max_pool2d(x['L-CC'], kernel_size=stride, stride=
            stride, padding=padding), 'L-MLO': F.max_pool2d(x['L-MLO'],
            kernel_size=stride, stride=stride, padding=padding), 'R-CC': F.
            max_pool2d(x['R-CC'], kernel_size=stride, stride=stride,
            padding=padding), 'R-MLO': F.max_pool2d(x['R-MLO'], kernel_size
            =stride, stride=stride, padding=padding)}


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {'L-CC': self._avg_pool(x['L-CC']), 'L-MLO': self._avg_pool(
            x['L-MLO']), 'R-CC': self._avg_pool(x['R-CC']), 'R-MLO': self.
            _avg_pool(x['R-MLO'])}

    @staticmethod
    def _avg_pool(single_view):
        n, c, h, w = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class AllViewsPad(nn.Module):
    """Pad tensor across all 4 views"""

    def __init__(self):
        super(AllViewsPad, self).__init__()

    def forward(self, x, pad):
        return {'L-CC': F.pad(x['L-CC'], pad), 'L-MLO': F.pad(x['L-MLO'],
            pad), 'R-CC': F.pad(x['R-CC'], pad), 'R-MLO': F.pad(x['R-MLO'],
            pad)}


class BaselineBreastModel(nn.Module):

    def __init__(self, device, nodropout_probability=None,
        gaussian_noise_std=None):
        super(BaselineBreastModel, self).__init__()
        self.conv_layer_dict = col.OrderedDict()
        self.conv_layer_dict['conv1'] = layers.AllViewsConvLayer(1,
            number_of_filters=32, filter_size=(3, 3), stride=(2, 2))
        self.conv_layer_dict['conv2a'] = layers.AllViewsConvLayer(32,
            number_of_filters=64, filter_size=(3, 3), stride=(2, 2))
        self.conv_layer_dict['conv2b'] = layers.AllViewsConvLayer(64,
            number_of_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv2c'] = layers.AllViewsConvLayer(64,
            number_of_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv3a'] = layers.AllViewsConvLayer(64,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv3b'] = layers.AllViewsConvLayer(128,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv3c'] = layers.AllViewsConvLayer(128,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv4a'] = layers.AllViewsConvLayer(128,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv4b'] = layers.AllViewsConvLayer(128,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv4c'] = layers.AllViewsConvLayer(128,
            number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv5a'] = layers.AllViewsConvLayer(128,
            number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv5b'] = layers.AllViewsConvLayer(256,
            number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict['conv5c'] = layers.AllViewsConvLayer(256,
            number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self._conv_layer_ls = nn.ModuleList(self.conv_layer_dict.values())
        self.all_views_pad = layers.AllViewsPad()
        self.all_views_max_pool = layers.AllViewsMaxPool()
        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.fc1 = nn.Linear(256 * 4, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 4)
        self.gaussian_noise_layer = layers.AllViewsGaussianNoise(
            gaussian_noise_std, device=device)
        self.dropout = nn.Dropout(p=1 - nodropout_probability)

    def forward(self, x):
        x = self.gaussian_noise_layer(x)
        x = self.conv_layer_dict['conv1'](x)
        x = self.all_views_max_pool(x, stride=(3, 3))
        x = self.conv_layer_dict['conv2a'](x)
        x = self.conv_layer_dict['conv2b'](x)
        x = self.conv_layer_dict['conv2c'](x)
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict['conv3a'](x)
        x = self.conv_layer_dict['conv3b'](x)
        x = self.conv_layer_dict['conv3c'](x)
        x = self.all_views_pad(x, pad=(0, 1, 0, 0))
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict['conv4a'](x)
        x = self.conv_layer_dict['conv4b'](x)
        x = self.conv_layer_dict['conv4c'](x)
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict['conv5a'](x)
        x = self.conv_layer_dict['conv5b'](x)
        x = self.conv_layer_dict['conv5c'](x)
        x = self.all_views_avg_pool(x)
        x = torch.cat([x['L-CC'], x['R-CC'], x['L-MLO'], x['R-MLO']], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class BaselineHistogramModel(nn.Module):

    def __init__(self, num_bins):
        super(BaselineHistogramModel, self).__init__()
        self.fc1 = nn.Linear(num_bins * 4, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def param_dict(self):
        return dict(zip(['w0', 'b0', 'w1', 'b1'], self.parameters()))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_nyukat_breast_density_classifier(_paritybench_base):
    pass
    def test_000(self):
        self._check(BaselineHistogramModel(*[], **{'num_bins': 4}), [torch.rand([16, 16])], {})

