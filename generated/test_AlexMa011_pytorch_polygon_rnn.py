import sys
_module = sys.modules[__name__]
del sys
config_tools = _module
data = _module
generate_data = _module
model = _module
test = _module
train = _module
utils = _module
convlstm = _module
utils = _module
validate = _module

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


import torch


from torch import utils


from torch.utils.data.dataset import Dataset


from torchvision import transforms


import torch.nn.init


import torch.utils.model_zoo as model_zoo


from torch import nn


import torch.utils.data


from torch.autograd import Variable


from torch import optim


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)), Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size])):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PolygonNet(nn.Module):

    def __init__(self, load_vgg=True):
        super(PolygonNet, self).__init__()

        def _make_basic(input_size, output_size, kernel_size, stride, padding):
            """

            :rtype: nn.Sequential
            """
            return nn.Sequential(nn.Conv2d(input_size, output_size, kernel_size, stride, padding), nn.ReLU(), nn.BatchNorm2d(output_size))
        self.model1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.model2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.model3 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.model4 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.convlayer1 = _make_basic(128, 128, 3, 1, 1)
        self.convlayer2 = _make_basic(256, 128, 3, 1, 1)
        self.convlayer3 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer4 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer5 = _make_basic(512, 128, 3, 1, 1)
        self.poollayer = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convlstm = ConvLSTM(input_size=(28, 28), input_dim=131, hidden_dim=[32, 8], kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=True)
        self.lstmlayer = nn.LSTM(28 * 28 * 8 + (28 * 28 + 3) * 2, 28 * 28 * 2, batch_first=True)
        self.linear = nn.Linear(28 * 28 * 2, 28 * 28 + 3)
        self.init_weights(load_vgg=load_vgg)

    def init_weights(self, load_vgg=True):
        """
        Initialize weights of PolygonNet
        :param load_vgg: bool
                    load pretrained vgg model or not
        """
        for name, param in self.convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.lstmlayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
        vgg_file = Path('vgg16_bn-6c64b313.pth')
        if load_vgg:
            if vgg_file.is_file():
                vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
            else:
                try:
                    wget.download('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
                    vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
                except:
                    vgg16_dict = torch.load(model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'))
            vgg_name = []
            for name in vgg16_dict:
                if 'feature' in name and 'running' not in name:
                    vgg_name.append(name)
            cnt = 0
            for name, param in self.named_parameters():
                if 'model' in name:
                    param.data.copy_(vgg16_dict[vgg_name[cnt]])
                    cnt += 1

    def forward(self, input_data1, first, second, third):
        bs = second.shape[0]
        length_s = second.shape[1]
        output1 = self.model1(input_data1)
        output11 = self.poollayer(output1)
        output11 = self.convlayer1(output11)
        output2 = self.model2(output1)
        output22 = self.convlayer2(output2)
        output3 = self.model3(output2)
        output33 = self.convlayer3(output3)
        output4 = self.model4(output3)
        output44 = self.convlayer4(output4)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        output = self.convlayer5(output)
        output = output.unsqueeze(1)
        output = output.repeat(1, length_s, 1, 1, 1)
        padding_f = torch.zeros([bs, 1, 1, 28, 28])
        input_f = first[:, :-3].view(-1, 1, 28, 28).unsqueeze(1).repeat(1, length_s - 1, 1, 1, 1)
        input_f = torch.cat([padding_f, input_f], dim=1)
        input_s = second[:, :, :-3].view(-1, length_s, 1, 28, 28)
        input_t = third[:, :, :-3].view(-1, length_s, 1, 28, 28)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)
        output = self.convlstm(output)[0][-1]
        shape_o = output.shape
        output = output.contiguous().view(bs, length_s, -1)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstmlayer(output)[0]
        output = output.contiguous().view(bs * length_s, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, length_s, -1)
        return output

    def test(self, input_data1, len_s):
        bs = input_data1.shape[0]
        result = torch.zeros([bs, len_s])
        output1 = self.model1(input_data1)
        output11 = self.poollayer(output1)
        output11 = self.convlayer1(output11)
        output2 = self.model2(output1)
        output22 = self.convlayer2(output2)
        output3 = self.model3(output2)
        output33 = self.convlayer3(output3)
        output4 = self.model4(output3)
        output44 = self.convlayer4(output4)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        feature = self.convlayer5(output)
        padding_f = torch.zeros([bs, 1, 1, 28, 28]).float()
        input_s = torch.zeros([bs, 1, 1, 28, 28]).float()
        input_t = torch.zeros([bs, 1, 1, 28, 28]).float()
        output = torch.cat([feature.unsqueeze(1), padding_f, input_s, input_t], dim=2)
        output, hidden1 = self.convlstm(output)
        output = output[-1]
        output = output.contiguous().view(bs, 1, -1)
        second = torch.zeros([bs, 1, 28 * 28 + 3])
        second[:, 0, 28 * 28 + 1] = 1
        third = torch.zeros([bs, 1, 28 * 28 + 3])
        third[:, 0, 28 * 28 + 2] = 1
        output = torch.cat([output, second, third], dim=2)
        output, hidden2 = self.lstmlayer(output)
        output = output.contiguous().view(bs, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, 1, -1)
        output = (output == output.max(dim=2, keepdim=True)[0]).float()
        first = output
        result[:, 0] = output.argmax(2)[:, 0]
        for i in range(len_s - 1):
            second = third
            third = output
            input_f = first[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_s = second[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_t = third[:, :, :-3].view(-1, 1, 1, 28, 28)
            input1 = torch.cat([feature.unsqueeze(1), input_f, input_s, input_t], dim=2)
            output, hidden1 = self.convlstm(input1, hidden1)
            output = output[-1]
            output = output.contiguous().view(bs, 1, -1)
            output = torch.cat([output, second, third], dim=2)
            output, hidden2 = self.lstmlayer(output, hidden2)
            output = output.contiguous().view(bs, -1)
            output = self.linear(output)
            output = output.contiguous().view(bs, 1, -1)
            output = (output == output.max(dim=2, keepdim=True)[0]).float()
            result[:, i + 1] = output.argmax(2)[:, 0]
        return result

