import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
setmodules = _module

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


import logging


import random


from collections import namedtuple


import time


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision.datasets


import torchvision.transforms


import numpy as np


import math


import torchvision.transforms as tt


import torchvision.transforms.functional as F1


from torch.nn.utils.rnn import pad_sequence


from torch.nn import Parameter


from copy import deepcopy


class PCAE(nn.Module):

    def __init__(self, config, num_capsules=24, template_size=11, num_templates=24, num_feature_maps=24):
        super(PCAE, self).__init__()
        self.num_capsules = num_capsules
        self.num_feature_maps = num_feature_maps
        self.capsules = nn.Sequential(nn.Conv2d(1, 128, 3, stride=2), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=2), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1), nn.ReLU(), nn.Conv2d(128, num_capsules * num_feature_maps, 1, stride=1))
        self.templates = nn.ParameterList([nn.Parameter(torch.randn(1, template_size, template_size)) for _ in range(num_templates)])
        self.soft_max = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to_pil = tt.ToPILImage()
        self.to_tensor = tt.ToTensor()
        self.epsilon = torch.tensor(1e-06)

    def forward(self, x, device, mode='train'):
        outputs = self.capsules(x)
        outputs = outputs.view(-1, self.num_capsules, self.num_feature_maps, *outputs.size()[2:])
        attention = outputs[:, :, -1, :, :].unsqueeze(2)
        attention_soft = self.soft_max(attention.view(*attention.size()[:3], -1)).view_as(attention)
        feature_maps = outputs[:, :, :-1, :, :]
        part_capsule_param = torch.sum(torch.sum(feature_maps * attention_soft, dim=-1), dim=-1)
        if mode == 'train':
            noise_1 = torch.FloatTensor(*part_capsule_param.size()[:2]).uniform_(-2, 2)
        else:
            noise_1 = torch.zeros(*part_capsule_param.size()[:2])
        x_m, d_m, c_z = self.relu(part_capsule_param[:, :, :6]), self.sigmoid(part_capsule_param[:, :, 6] + noise_1).view(*part_capsule_param.size()[:2], 1), self.relu(part_capsule_param[:, :, 7:])
        B, _, _, target_size = x.size()
        transformed_templates = [F.grid_sample(self.templates[i].repeat(B, 1, 1, 1), F.affine_grid(self.geometric_transform(x_m[:, i, :]), torch.Size((B, 1, target_size, target_size)))) for i in range(self.num_capsules)]
        transformed_templates = torch.cat(transformed_templates, 1)
        mix_prob = self.soft_max(d_m * transformed_templates.view(*transformed_templates.size()[:2], -1)).view_as(transformed_templates)
        detach_x = x.data
        std = detach_x.view(*x.size()[:2], -1).std(-1).unsqueeze(1)
        std = std * 1 + self.epsilon
        multiplier = (std * math.sqrt(math.pi * 2)).reciprocal().unsqueeze(-1)
        power_multiply = (-(2 * std ** 2)).reciprocal().unsqueeze(-1)
        gaussians = multiplier * ((detach_x - transformed_templates) ** 2 * power_multiply).exp()
        pre_ll = gaussians * mix_prob * 1.0 + self.epsilon
        log_likelihood = torch.sum(pre_ll, dim=1).log().sum(-1).sum(-1).mean()
        x_m_detach = x_m.data
        d_m_detach = d_m.data
        template_det = []
        for template in self.templates:
            template_det.append(template.data.view(1, -1))
        template_detached = torch.cat(template_det, 0).unsqueeze(0).expand(x_m_detach.shape[0], -1, -1)
        input_ocae = torch.cat([d_m_detach, x_m_detach, template_detached, c_z], -1)
        return log_likelihood, input_ocae, x_m_detach, d_m_detach

    @staticmethod
    def geometric_transform(pose_tensor, similarity=False, nonlinear=True):
        """Convers paramer tensor into an affine or similarity transform.
        This function is adapted from:
        https://github.com/akosiorek/stacked_capsule_autoencoders/blob/master/capsules/math_ops.py

        Args:
        pose_tensor: [..., 6] tensor.
        similarity: bool.
        nonlinear: bool; applies nonlinearities to pose params if True.

        Returns:
        [..., 2, 3] tensor.
        """
        scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor, 1, -1)
        if nonlinear:
            scale_x, scale_y = torch.sigmoid(scale_x) + 0.01, torch.sigmoid(scale_y) + 0.01
            trans_x, trans_y, shear = torch.tanh(trans_x * 5.0), torch.tanh(trans_y * 5.0), torch.tanh(shear * 5.0)
            theta *= 2.0 * math.pi
        else:
            scale_x, scale_y = (abs(i) + 0.01 for i in (scale_x, scale_y))
        c, s = torch.cos(theta), torch.sin(theta)
        if similarity:
            scale = scale_x
            pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]
        else:
            pose = [scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c, trans_x, scale_y * s, scale_y * c, trans_y]
        pose = torch.cat(pose, -1)
        shape = list(pose.shape[:-1])
        shape += [2, 3]
        pose = torch.reshape(pose, shape)
        return pose


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class SetTransformer(nn.Module):

    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln), ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(PMA(dim_hidden, num_heads, num_outputs, ln=ln), SAB(dim_hidden, dim_hidden, num_heads, ln=ln), SAB(dim_hidden, dim_hidden, num_heads, ln=ln), nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class OCAE(nn.Module):

    def __init__(self, config, dim_input=144, num_capsules=24, set_out=256, set_head=1, special_feat=16):
        super(OCAE, self).__init__()
        self.set_transformer = nn.Sequential(SetTransformer(dim_input, num_capsules, set_out, num_heads=set_head, dim_hidden=16, ln=True), SetTransformer(set_out, num_capsules, set_out, num_heads=set_head, dim_hidden=16, ln=True), SetTransformer(set_out, num_capsules, special_feat + 1 + 9, num_heads=set_head, dim_hidden=16, ln=True))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(special_feat, special_feat), nn.ReLU(), nn.Linear(special_feat, 48)) for _ in range(num_capsules)])
        self.op_mat = Parameter(torch.randn(num_capsules, num_capsules, 3, 3))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.epsilon = torch.tensor(1e-06)

    def forward(self, inp, x_m, d_m, device, mode='train'):
        object_parts = self.set_transformer(inp)
        if mode == 'train':
            noise_1 = torch.FloatTensor(*object_parts.size()[:2]).uniform_(-2, 2)
            noise_2 = torch.FloatTensor(object_parts.shape[0], 24).uniform_(-2, 2)
        else:
            noise_1 = torch.zeros(*object_parts.size()[:2])
            noise_2 = torch.zeros(object_parts.shape[0], 24)
        ov_k, c_k, a_k = self.relu(object_parts[:, :, :9]).view(*object_parts.size()[:2], 1, 3, 3), self.relu(object_parts[:, :, 9:25]), self.sigmoid(object_parts[:, :, -1] + noise_1).view(*object_parts.size()[:2], 1, 1, 1)
        temp_a = []
        temp_lambda = []
        for num, mlp in enumerate(self.mlps):
            mlp_out = self.mlps[num](c_k[:, num, :])
            temp_a.append(self.sigmoid(mlp_out[:, :24] + noise_2).unsqueeze(1))
            temp_lambda.append(self.relu(mlp_out[:, 24:]).unsqueeze(1))
        a_kn = torch.cat(temp_a, 1).unsqueeze(-1).unsqueeze(-1)
        lambda_kn = torch.cat(temp_lambda, 1).unsqueeze(-1).unsqueeze(-1)
        lambda_kn = lambda_kn * 1 + self.epsilon
        v_kn = ov_k.matmul(self.op_mat)
        mu_kn = v_kn.view(*v_kn.size()[:3], -1)[:, :, :, :6]
        x_m = x_m.unsqueeze(1)
        diff = (x_m - mu_kn).unsqueeze(-2)
        identity = torch.eye(6).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*diff.size()[:3], -1, -1)
        cov_matrix_inv = lambda_kn.reciprocal() * identity
        mahalanobis = torch.matmul(torch.matmul(diff, cov_matrix_inv), diff.transpose(-1, -2))
        gaussian_multiplier = ((2 * math.pi) ** 6 * lambda_kn ** 6).sqrt()
        gaussian = (-0.5 * mahalanobis).exp() * gaussian_multiplier.reciprocal()
        gaussian_component = a_k * a_kn * (a_k.sum(1).unsqueeze(1) * a_kn.sum(2).unsqueeze(1)).reciprocal()
        gauss_mix = (gaussian * gaussian_component).squeeze(-1).squeeze(-1)
        gauss_mix = gauss_mix * 1.0 + self.epsilon
        before_log = gauss_mix.sum(1).log()
        log_likelihood = (before_log * d_m.view(before_log.shape[0], -1)).sum(-1).mean()
        return log_likelihood, a_k.squeeze(-1).squeeze(-1), a_kn.squeeze(-1).squeeze(-1), gaussian.squeeze(-1).squeeze(-1)


class SCAE(nn.Module):

    def __init__(self, config=None):
        super(SCAE, self).__init__()
        self.pcae = PCAE(config)
        self.ocae = OCAE(config)

    def forward(self, x, device, mode):
        image_likelihood, input_ocae, x_m, d_m = self.pcae(x, device, mode)
        part_likelihood, a_k, a_kn, gaussian = self.ocae(input_ocae, x_m, d_m, device, mode)
        return image_likelihood, part_likelihood, a_k, a_kn, gaussian


class SCAE_LOSS(nn.Module):

    def __init__(self):
        super(SCAE_LOSS, self).__init__()

    def entropy(self, x):
        h = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
        h = -1.0 * h.sum(-1)
        return h.mean()

    def forward(self, output_scae, b_c, k_c):
        img_lik, part_lik, a_k, a_kn, gaussian = output_scae
        a_k_prior = a_k.squeeze(-1) * a_kn.max(-1).values
        a_kn_posterior = a_k * (a_kn * gaussian)
        l_11 = (a_k_prior.sum(-1) - k_c).pow(2).mean()
        l_12 = (a_k_prior.sum(0) - b_c).pow(2).mean()
        prior_sparsity = l_11 + l_12
        v_k = a_kn_posterior.sum(-1).transpose(0, 1)
        v_b = a_kn_posterior.sum(-1)
        posterior_sparsity = self.entropy(v_k) - self.entropy(v_b)
        return -img_lik - part_lik + prior_sparsity + 10 * posterior_sparsity


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ISAB,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'num_heads': 4, 'num_inds': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MAB,
     lambda: ([], {'dim_Q': 4, 'dim_K': 4, 'dim_V': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PMA,
     lambda: ([], {'dim': 4, 'num_heads': 4, 'num_seeds': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SAB,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SCAE_LOSS,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SetTransformer,
     lambda: ([], {'dim_input': 4, 'num_outputs': 4, 'dim_output': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_phanideepgampa_stacked_capsule_networks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

