import sys
_module = sys.modules[__name__]
del sys
regression_losses = _module

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


import warnings


import time


import copy


import numpy as np


import torch


from torch.utils.data import DataLoader


from torchvision.utils import save_image


from torchvision import datasets


from torchvision import transforms


from matplotlib import pyplot


class LogCoshLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class AlgebraicLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t))


DEVICE = 'cpu'


class MNISTAutoencoderModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._monitor = {'elapse_total_ms': 0, 'learning': {'losses': []}, 'testing': {'losses': []}}
        self._scheduler = None
        self._optim = None
        self._criterion = None
        self._encoder_fc, self._latent_fc, self._decoder_fc = self._construct()

    def forward(self, x_t):
        x_t = self._encoder_fc(x_t)
        x_t = self._latent_fc(x_t)
        x_t = self._decoder_fc(x_t)
        return x_t

    @staticmethod
    def _construct():
        encoder_seqs = []
        latent_seqs = []
        decoder_seqs = []
        encoder_fclayer1 = torch.nn.Linear(in_features=28 * 28, out_features=256)
        torch.nn.init.xavier_normal_(encoder_fclayer1.weight)
        encoder_fclayer1.bias.data.fill_(0.0)
        encoder_seqs.append(encoder_fclayer1)
        encoder_seqs.append(torch.nn.ReLU(inplace=True))
        encoder_fclayer2 = torch.nn.Linear(in_features=256, out_features=64)
        torch.nn.init.xavier_normal_(encoder_fclayer2.weight)
        encoder_fclayer2.bias.data.fill_(0.0)
        encoder_seqs.append(encoder_fclayer2)
        encoder_seqs.append(torch.nn.ReLU(inplace=True))
        latent_fclayer1 = torch.nn.Linear(in_features=64, out_features=64)
        torch.nn.init.xavier_normal_(latent_fclayer1.weight)
        latent_fclayer1.bias.data.fill_(0.0)
        latent_seqs.append(latent_fclayer1)
        latent_seqs.append(torch.nn.BatchNorm1d(num_features=64))
        latent_seqs.append(torch.nn.ReLU(inplace=True))
        decoder_fclayer2 = torch.nn.Linear(in_features=64, out_features=256)
        torch.nn.init.xavier_normal_(decoder_fclayer2.weight)
        decoder_fclayer2.bias.data.fill_(0.0)
        decoder_seqs.append(decoder_fclayer2)
        decoder_seqs.append(torch.nn.ReLU(inplace=True))
        decoder_fclayer1 = torch.nn.Linear(in_features=256, out_features=28 * 28)
        torch.nn.init.xavier_normal_(decoder_fclayer1.weight)
        decoder_fclayer1.bias.data.fill_(0.0)
        decoder_seqs.append(decoder_fclayer1)
        decoder_seqs.append(torch.nn.Tanh())
        return torch.nn.Sequential(*encoder_seqs), torch.nn.Sequential(*latent_seqs), torch.nn.Sequential(*decoder_seqs)

    @property
    def monitor(self):
        return copy.deepcopy(self._monitor)

    def setup(self, *, criterion='mse', optim='adam', lr=0.001):
        if isinstance(optim, str):
            if optim == 'sgd':
                self._optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
            elif optim == 'sgdm':
                self._optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            elif optim == 'adam':
                self._optim = torch.optim.Adam(self.parameters(), lr=lr)
            else:
                raise TypeError('Unknown optimizer type %s.' % optim)
        else:
            self._optim = optim
        if isinstance(criterion, str):
            if criterion == 'mse' or criterion == 'mean_square_error':
                self._criterion = torch.nn.MSELoss(reduction='mean')
            elif criterion == 'mae' or criterion == 'mean_absolute_error':
                self._criterion = torch.nn.L1Loss(reduction='mean')
            elif criterion == 'xtl' or criterion == 'xtanh_loss':
                self._criterion = XTanhLoss()
            elif criterion == 'xsl' or criterion == 'xsigmoid_loss':
                self._criterion = XSigmoidLoss()
            elif criterion == 'agl' or criterion == 'algebraic_loss':
                self._criterion = AlgebraicLoss()
            elif criterion == 'lcl' or criterion == 'log_cosh_loss':
                self._criterion = LogCoshLoss()
            else:
                raise TypeError('Unknown criterion type %s.' % criterion)
            self._scheduler = torch.optim.lr_scheduler.StepLR(self._optim, step_size=5, gamma=0.9)
        else:
            self._criterion = criterion

    def infer(self, x_t):
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).float()
        else:
            x_t = x_t
        with torch.no_grad():
            y_t = self(x_t)
        return y_t

    def learn(self, x_t, y_prime_t, *, epoch_limit=50, batch_size=32, tl_split=0.2):
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t).float()
        else:
            x_t = x_t
        if isinstance(y_prime_t, np.ndarray):
            y_prime_t = torch.from_numpy(y_prime_t).float()
        else:
            y_prime_t = y_prime_t
        input_sample_size = x_t.shape[0]
        expected_output_sample_size = y_prime_t.shape[0]
        if input_sample_size != expected_output_sample_size:
            warnings.warn('Input training dataset is not the same lenght as the expected output dataset.', UserWarning)
        self._monitor['elapse_total_ms'] = 0
        self._monitor['learning']['losses'] = []
        self._monitor['testing']['losses'] = []
        if tl_split < 0 or tl_split > 0.5:
            tl_split = 0
            warnings.warn('Testing and learning split ratio must be >= 0 and <= 0.5. Reset testing and learning split ratio to default value of 0.', UserWarning)
        enable_testing = tl_split > 0
        if enable_testing:
            if input_sample_size == 1:
                learning_sample_size = input_sample_size
                enable_testing = False
                warnings.warn('Input sample size = 1. Reset testing and learning split ratio to default value of 0.', UserWarning)
            else:
                learning_sample_size = int(input_sample_size * (1 - tl_split))
                learning_sample_size = learning_sample_size - learning_sample_size % batch_size
        else:
            learning_sample_size = input_sample_size
        if batch_size < 1 or batch_size > learning_sample_size:
            batch_size = learning_sample_size
            warnings.warn('Batch size must be >= 1 and <= learning sample size %d. Set batch size = learning sample size.' % learning_sample_size, UserWarning)
        elapse_total_ms = 0
        for epoch in range(epoch_limit):
            tstart_us = time.process_time()
            learning_loss_value = 0
            testing_loss_value = 0
            for i in range(learning_sample_size):
                if i + batch_size < learning_sample_size:
                    batched_x_t = x_t[i:i + batch_size].requires_grad_(True)
                    batched_y_prime_t = y_prime_t[i:i + batch_size].requires_grad_(False)
                batched_y_t = self(batched_x_t)
                learning_loss_t = self._criterion(batched_y_t, batched_y_prime_t)
                self._optim.zero_grad()
                learning_loss_t.backward()
                self._optim.step()
                learning_loss_value += learning_loss_t.item()
            learning_loss_value /= learning_sample_size
            self._monitor['learning']['losses'].append(learning_loss_value)
            if enable_testing:
                with torch.no_grad():
                    batched_x_t = x_t[learning_sample_size:].requires_grad_(False)
                    batched_y_prime_t = y_prime_t[learning_sample_size:].requires_grad_(False)
                    batched_y_t = self(batched_x_t)
                    testing_loss_t = self._criterion(batched_y_t, batched_y_prime_t)
                    testing_loss_value = testing_loss_t.item()
                    self._monitor['testing']['losses'].append(testing_loss_value)
            tend_us = time.process_time()
            elapse_per_pass_ms = int(round((tend_us - tstart_us) * 1000))
            elapse_total_ms += elapse_per_pass_ms
            self._monitor['elapse_total_ms'] = elapse_total_ms
            if self._scheduler is not None:
                self._scheduler.step()
                lr = self._scheduler.get_lr()[0]
                None
            if enable_testing:
                None
            else:
                None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlgebraicLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogCoshLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (XSigmoidLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (XTanhLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tuantle_regression_losses_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

