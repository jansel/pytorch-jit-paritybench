import sys
_module = sys.modules[__name__]
del sys
master = _module
dataloader = _module
main = _module
mnist_bags_loader = _module
model = _module

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


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.feature_extractor_part1 = nn.Sequential(nn.Conv2d(1, 20,
            kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d
            (20, 50, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.feature_extractor_part2 = nn.Sequential(nn.Linear(50 * 4 * 4,
            self.L), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(),
            nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.
            Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().data[0]
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-05, max=1.0 - 1e-05)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) *
            torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A


class GatedAttention(nn.Module):

    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.feature_extractor_part1 = nn.Sequential(nn.Conv2d(1, 20,
            kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d
            (20, 50, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.feature_extractor_part2 = nn.Sequential(nn.Linear(50 * 4 * 4,
            self.L), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.
            Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.
            Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-05, max=1.0 - 1e-05)
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) *
            torch.log(1.0 - Y_prob))
        return neg_log_likelihood, A


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_AMLab_Amsterdam_AttentionDeepMIL(_paritybench_base):
    pass
