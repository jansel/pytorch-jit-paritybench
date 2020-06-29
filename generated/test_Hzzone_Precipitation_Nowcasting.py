import sys
_module = sys.modules[__name__]
del sys
demo = _module
setup = _module
varflow = _module
benchmark = _module
main = _module
net_params = _module
rover_and_last_frame = _module
config = _module
gifmaker = _module
msssim = _module
ordered_easydict = _module
visualization = _module
dataloader = _module
evaluation = _module
image = _module
mask = _module
numba_accelerated = _module
convLSTM = _module
encoder = _module
forecaster = _module
loss = _module
model = _module
probToPixel = _module
trajGRU = _module
train_and_test = _module
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


from collections import OrderedDict


from torch.optim import lr_scheduler


import numpy as np


import copy


import time


from torch import nn


import logging


import torch.nn.functional as F


_global_config['GLOBAL'] = 4


_global_config['HKO'] = 4


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(
                    negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(
                    negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):

    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1),
            input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None)
        return outputs_stage, state_stage

    def forward(self, input):
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(input, getattr(self,
                'stage' + str(i)), getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):

    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(
                params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=cfg.HKO.BENCHMARK.
            OUT_LEN)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1),
            input.size(2), input.size(3)))
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self,
            'stage3'), getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i - 1],
                getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)))
        return input


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


class Weighted_mse_mae(nn.Module):

    def __init__(self, mse_weight=1.0, mae_weight=1.0,
        NORMAL_LOSS_GLOBAL_SCALE=5e-05, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.
            THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] -
                balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        mse = torch.sum(weights * (input - target) ** 2, (2, 3, 4))
        mae = torch.sum(weights * torch.abs(input - target), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight * torch.
            mean(mse) + self.mae_weight * torch.mean(mae))


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMBDA
        self._thresholds = thresholds

    def forward(self, input, target, mask):
        assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN
        input = input.permute((1, 2, 0, 3, 4))
        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + rainfall_to_pixel(self._thresholds).tolist()
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        error = F.cross_entropy(input, class_index, self._weight, reduction
            ='none')
        if self._lambda is not None:
            B, S, H, W = error.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        error = error.permute(1, 0, 2, 3).unsqueeze(2)
        return torch.mean(error * mask.float())


class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output


class Predictor(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.model = make_layers(params)

    def forward(self, input):
        """
        input: S*B*1*H*W
        :param input:
        :return:
        """
        input = input.squeeze(2).permute((1, 0, 2, 3))
        output = self.model(input)
        return output.unsqueeze(2).permute((1, 0, 2, 3, 4))


class BaseConvRNN(nn.Module):

    def __init__(self, num_filter, b_h_w, h2h_kernel=(3, 3), h2h_dilate=(1,
        1), i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
        i2h_dilate=(1, 1), act_type=torch.tanh, prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert self._h2h_kernel[0] % 2 == 1 and self._h2h_kernel[1
            ] % 2 == 1, 'Only support odd number, get h2h_kernel= %s' % str(
            h2h_kernel)
        self._h2h_pad = h2h_dilate[0] * (h2h_kernel[0] - 1) // 2, h2h_dilate[1
            ] * (h2h_kernel[1] - 1) // 2
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0
            ]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1
            ]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] -
            i2h_dilate_ksize_h) // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] -
            i2h_dilate_ksize_w) // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Hzzone_Precipitation_Nowcasting(_paritybench_base):
    pass
    def test_000(self):
        self._check(EF(*[], **{'encoder': ReLU(), 'forecaster': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

