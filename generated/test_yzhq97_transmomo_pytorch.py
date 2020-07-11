import sys
_module = sys.modules[__name__]
del sys
infer_pair = _module
lib = _module
data = _module
loss = _module
network = _module
operation = _module
trainer = _module
util = _module
general = _module
global_norm = _module
motion = _module
visualization = _module
render_interpolate = _module
scripts = _module
compute_mse = _module
preprocess = _module
rotate_test_set = _module
test = _module
train = _module

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


import torch.backends.cudnn as cudnn


from scipy.ndimage import gaussian_filter1d


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch.nn as nn


from math import pi


import logging


from torch.autograd import Variable


from torch.optim import lr_scheduler


import math


import torchvision.utils as vutils


import torch.nn.init as init


import time


from itertools import combinations


class ConvEncoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        conv_pool = None if config.conv_pool is None else getattr(nn, config.conv_pool)
        encoder = cls(config.channels, config.padding, config.kernel_size, config.conv_stride, conv_pool)
        return encoder

    def __init__(self, channels, padding=3, kernel_size=8, conv_stride=2, conv_pool=None):
        super(ConvEncoder, self).__init__()
        self.in_channels = channels[0]
        model = []
        acti = nn.LeakyReLU(0.2)
        nr_layer = len(channels) - 1
        for i in range(nr_layer):
            if conv_pool is None:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
            else:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
                model.append(conv_pool(kernel_size=2, stride=2))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x[:, :self.in_channels, :]
        x = self.model(x)
        return x


class ConvDecoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        decoder = cls(config.channels, config.kernel_size)
        return decoder

    def __init__(self, channels, kernel_size=7):
        super(ConvDecoder, self).__init__()
        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)
        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


thismodule = sys.modules[__name__]


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.gan_type = config.gan_type
        encoder_cls = getattr(thismodule, config.encoder_cls)
        self.encoder = encoder_cls.build_from_config(config)
        self.linear = nn.Linear(config.channels[-1], 1)

    def forward(self, seqs):
        code_seq = self.encoder(seqs)
        logits = self.linear(code_seq.permute(0, 2, 1))
        return logits

    def calc_dis_loss(self, x_gen, x_real):
        fake_logits = self.forward(x_gen)
        real_logits = self.forward(x_real)
        if self.gan_type == 'lsgan':
            loss = torch.mean((fake_logits - 0) ** 2) + torch.mean((real_logits - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(fake_logits, requires_grad=False)
            all1 = torch.ones_like(real_logits, requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(fake_logits), all0) + F.binary_cross_entropy(F.sigmoid(real_logits), all1))
        else:
            raise NotImplementedError
        return loss

    def calc_gen_loss(self, x_gen):
        logits = self.forward(x_gen)
        if self.gan_type == 'lsgan':
            loss = torch.mean((logits - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(logits, requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(logits), all1))
        else:
            raise NotImplementedError
        return loss


def change_of_basis(motion_3d, basis_vectors=None, project_2d=False):
    if basis_vectors is None:
        motion_proj = motion_3d[:, :, ([0, 2]), :]
    else:
        if project_2d:
            basis_vectors = basis_vectors[:, :, :, ([0, 2]), :]
        _, K, seq_len, _, _ = basis_vectors.size()
        motion_3d = motion_3d.unsqueeze(1).repeat(1, K, 1, 1, 1)
        motion_3d = motion_3d.permute([0, 1, 4, 3, 2])
        motion_proj = basis_vectors @ motion_3d
        motion_proj = motion_proj.permute([0, 1, 4, 3, 2])
    return motion_proj


def get_body_basis(motion_3d):
    """
    Get the unit vectors for vector rectangular coordinates for given 3D motion
    :param motion_3d: 3D motion from 3D joints positions, shape (B, n_joints, 3, seq_len).
    :param angles: (K, 3), Rotation angles around each axis.
    :return: unit vectors for vector rectangular coordinates's , shape (B, 3, 3).
    """
    B = motion_3d.size(0)
    horizontal = (motion_3d[:, (2)] - motion_3d[:, (5)] + motion_3d[:, (9)] - motion_3d[:, (12)]) / 2
    horizontal = horizontal.mean(dim=-1)
    horizontal = horizontal / horizontal.norm(dim=-1).unsqueeze(-1)
    vector_z = torch.tensor([0.0, 0.0, 1.0], device=motion_3d.device, dtype=motion_3d.dtype).unsqueeze(0).repeat(B, 1)
    vector_y = torch.cross(horizontal, vector_z)
    vector_y = vector_y / vector_y.norm(dim=-1).unsqueeze(-1)
    vector_x = torch.cross(vector_y, vector_z)
    vectors = torch.stack([vector_x, vector_y, vector_z], dim=2)
    vectors = vectors.detach()
    return vectors


def rotate_basis_euler(basis_vectors, angles):
    """
    Rotate vector rectangular coordinates from given angles.

    :param basis_vectors: [B, 3, 3]
    :param angles: [B, K, T, 3] Rotation angles around each axis.
    :return: [B, K, T, 3, 3]
    """
    B, K, T, _ = angles.size()
    cos, sin = torch.cos(angles), torch.sin(angles)
    cx, cy, cz = cos[:, :, :, (0)], cos[:, :, :, (1)], cos[:, :, :, (2)]
    sx, sy, sz = sin[:, :, :, (0)], sin[:, :, :, (1)], sin[:, :, :, (2)]
    x = basis_vectors[:, (0), :]
    o = torch.zeros_like(x[:, (0)])
    x_cpm_0 = torch.stack([o, -x[:, (2)], x[:, (1)]], dim=1)
    x_cpm_1 = torch.stack([x[:, (2)], o, -x[:, (0)]], dim=1)
    x_cpm_2 = torch.stack([-x[:, (1)], x[:, (0)], o], dim=1)
    x_cpm = torch.stack([x_cpm_0, x_cpm_1, x_cpm_2], dim=1)
    x_cpm = x_cpm.unsqueeze(1).unsqueeze(2)
    x = x.unsqueeze(-1)
    xx = torch.matmul(x, x.transpose(-1, -2)).unsqueeze(1).unsqueeze(2)
    eye = torch.eye(n=3, dtype=basis_vectors.dtype, device=basis_vectors.device)
    eye = eye.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mat33_x = cx.unsqueeze(-1).unsqueeze(-1) * eye + sx.unsqueeze(-1).unsqueeze(-1) * x_cpm + (1.0 - cx).unsqueeze(-1).unsqueeze(-1) * xx
    o = torch.zeros_like(cz)
    i = torch.ones_like(cz)
    mat33_z_0 = torch.stack([cz, sz, o], dim=3)
    mat33_z_1 = torch.stack([-sz, cz, o], dim=3)
    mat33_z_2 = torch.stack([o, o, i], dim=3)
    mat33_z = torch.stack([mat33_z_0, mat33_z_1, mat33_z_2], dim=3)
    basis_vectors = basis_vectors.unsqueeze(1).unsqueeze(2)
    basis_vectors = basis_vectors @ mat33_x.transpose(-1, -2) @ mat33_z
    return basis_vectors


def rotate_and_maybe_project(X, angles=None, body_reference=True, project_2d=False):
    D = 2 if project_2d else 3
    batch_size, channels, seq_len = X.size()
    n_joints = channels // 3
    X = X.view(batch_size, n_joints, 3, seq_len)
    if angles is not None:
        K = angles.size(1)
        basis_vectors = get_body_basis(X) if body_reference else torch.eye(3, device=X.device).unsqueeze(0).repeat(batch_size, 1, 1)
        basis_vectors = rotate_basis_euler(basis_vectors, angles)
        X_trans = change_of_basis(X, basis_vectors, project_2d=project_2d)
        X_trans = X_trans.reshape(batch_size * K, n_joints * D, seq_len)
    else:
        X_trans = change_of_basis(X, project_2d=project_2d)
        X_trans = X_trans.reshape(batch_size, n_joints * D, seq_len)
    return X_trans


class Autoencoder3f(nn.Module):

    def __init__(self, config):
        super(Autoencoder3f, self).__init__()
        assert config.motion_encoder.channels[-1] + config.body_encoder.channels[-1] + config.view_encoder.channels[-1] == config.decoder.channels[0]
        self.n_joints = config.decoder.channels[-1] // 3
        self.body_reference = config.body_reference
        motion_cls = getattr(thismodule, config.motion_encoder.cls)
        body_cls = getattr(thismodule, config.body_encoder.cls)
        view_cls = getattr(thismodule, config.view_encoder.cls)
        self.motion_encoder = motion_cls.build_from_config(config.motion_encoder)
        self.body_encoder = body_cls.build_from_config(config.body_encoder)
        self.view_encoder = view_cls.build_from_config(config.view_encoder)
        self.decoder = ConvDecoder.build_from_config(config.decoder)
        self.body_pool = getattr(F, config.body_encoder.global_pool) if config.body_encoder.global_pool is not None else None
        self.view_pool = getattr(F, config.view_encoder.global_pool) if config.view_encoder.global_pool is not None else None

    def forward(self, seqs):
        return self.reconstruct(seqs)

    def encode_motion(self, seqs):
        motion_code_seq = self.motion_encoder(seqs)
        return motion_code_seq

    def encode_body(self, seqs):
        body_code_seq = self.body_encoder(seqs)
        kernel_size = body_code_seq.size(-1)
        body_code = self.body_pool(body_code_seq, kernel_size) if self.body_pool is not None else body_code_seq
        return body_code, body_code_seq

    def encode_view(self, seqs):
        view_code_seq = self.view_encoder(seqs)
        kernel_size = view_code_seq.size(-1)
        view_code = self.view_pool(view_code_seq, kernel_size) if self.view_pool is not None else view_code_seq
        return view_code, view_code_seq

    def decode(self, motion_code, body_code, view_code):
        if body_code.size(-1) == 1:
            body_code = body_code.repeat(1, 1, motion_code.shape[-1])
        if view_code.size(-1) == 1:
            view_code = view_code.repeat(1, 1, motion_code.shape[-1])
        complete_code = torch.cat([motion_code, body_code, view_code], dim=1)
        out = self.decoder(complete_code)
        return out

    def cross3d(self, x_a, x_b, x_c):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_b)
        view_c, _ = self.encode_view(x_c)
        out = self.decode(motion_a, body_b, view_c)
        return out

    def cross2d(self, x_a, x_b, x_c):
        motion_a = self.encode_motion(x_a)
        body_b, _ = self.encode_body(x_b)
        view_c, _ = self.encode_view(x_c)
        out = self.decode(motion_a, body_b, view_c)
        out = rotate_and_maybe_project(out, body_reference=self.body_reference, project_2d=True)
        return out

    def reconstruct3d(self, x):
        motion_code = self.encode_motion(x)
        body_code, _ = self.encode_body(x)
        view_code, _ = self.encode_view(x)
        out = self.decode(motion_code, body_code, view_code)
        return out

    def reconstruct2d(self, x):
        motion_code = self.encode_motion(x)
        body_code, _ = self.encode_body(x)
        view_code, _ = self.encode_view(x)
        out = self.decode(motion_code, body_code, view_code)
        out = rotate_and_maybe_project(out, body_reference=self.body_reference, project_2d=True)
        return out

    def interpolate(self, x_a, x_b, N):
        step_size = 1.0 / (N - 1)
        batch_size, _, seq_len = x_a.size()
        motion_a = self.encode_motion(x_a)
        body_a, body_a_seq = self.encode_body(x_a)
        view_a, view_a_seq = self.encode_view(x_a)
        motion_b = self.encode_motion(x_b)
        body_b, body_b_seq = self.encode_body(x_b)
        view_b, view_b_seq = self.encode_view(x_b)
        batch_out = torch.zeros([batch_size, N, N, 2 * self.n_joints, seq_len])
        for i in range(N):
            motion_weight = i * step_size
            for j in range(N):
                body_weight = j * step_size
                motion = (1.0 - motion_weight) * motion_a + motion_weight * motion_b
                body = (1.0 - body_weight) * body_a + body_weight * body_b
                view = (1.0 - body_weight) * view_a + body_weight * view_b
                out = self.decode(motion, body, view)
                out = rotate_and_maybe_project(out, body_reference=self.body_reference, project_2d=True)
                batch_out[:, (i), (j), :, :] = out
        return batch_out


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and key in f and '.pt' in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'], gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class BaseTrainer(nn.Module):

    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        lr = config.lr
        autoencoder_cls = getattr(lib.network, config.autoencoder.cls)
        self.autoencoder = autoencoder_cls(config.autoencoder)
        self.discriminator = Discriminator(config.discriminator)
        beta1 = config.beta1
        beta2 = config.beta2
        dis_params = list(self.discriminator.parameters())
        ae_params = list(self.autoencoder.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.ae_opt = torch.optim.Adam([p for p in ae_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.ae_scheduler = get_scheduler(self.ae_opt, config)
        self.apply(weights_init(config.init))
        self.discriminator.apply(weights_init('gaussian'))

    def forward(self, data):
        x_a, x_b = data['x_a'], data['x_b']
        batch_size = x_a.size(0)
        self.eval()
        body_a, body_b = self.sample_body_code(batch_size)
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a_enc, _ = self.autoencoder.encode_body(x_a)
        motion_b = self.autoencoder.encode_motion(x_b)
        body_b_enc, _ = self.autoencoder.encode_body(x_b)
        x_ab = self.autoencoder.decode(motion_a, body_b)
        x_ba = self.autoencoder.decode(motion_b, body_a)
        self.train()
        return x_ab, x_ba

    def dis_update(self, data, config):
        raise NotImplemented

    def ae_update(self, data, config):
        raise NotImplemented

    def recon_criterion(self, input, target):
        raise NotImplemented

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.ae_scheduler is not None:
            self.ae_scheduler.step()

    def resume(self, checkpoint_dir, config):
        last_model_name = get_model_list(checkpoint_dir, 'autoencoder')
        state_dict = torch.load(last_model_name)
        self.autoencoder.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])
        last_model_name = get_model_list(checkpoint_dir, 'discriminator')
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['discriminator'])
        self.ae_opt.load_state_dict(state_dict['autoencoder'])
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.ae_scheduler = get_scheduler(self.ae_opt, config, iterations)
        None
        return iterations

    def save(self, snapshot_dir, iterations):
        ae_name = os.path.join(snapshot_dir, 'autoencoder_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'discriminator_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(self.autoencoder.state_dict(), ae_name)
        torch.save(self.discriminator.state_dict(), dis_name)
        torch.save({'autoencoder': self.ae_opt.state_dict(), 'discriminator': self.dis_opt.state_dict()}, opt_name)

    def validate(self, data, config):
        re_dict = self.evaluate(self.autoencoder, data, config)
        for key, val in re_dict.items():
            setattr(self, key, val)

    @staticmethod
    def recon_criterion(input, target):
        return torch.mean(torch.abs(input - target))

    @classmethod
    def evaluate(cls, autoencoder, data, config):
        autoencoder.eval()
        x_a, x_b = data['x_a'], data['x_b']
        x_aba, x_bab = data['x_aba'], data['x_bab']
        batch_size, _, seq_len = x_a.size()
        re_dict = {}
        with torch.no_grad():
            x_a_recon = autoencoder.reconstruct2d(x_a)
            x_b_recon = autoencoder.reconstruct2d(x_b)
            x_aba_recon = autoencoder.cross2d(x_a, x_b, x_a)
            x_bab_recon = autoencoder.cross2d(x_b, x_a, x_b)
            re_dict['loss_val_recon_x'] = cls.recon_criterion(x_a_recon, x_a) + cls.recon_criterion(x_b_recon, x_b)
            re_dict['loss_val_cross_body'] = cls.recon_criterion(x_aba_recon, x_aba) + cls.recon_criterion(x_bab_recon, x_bab)
            re_dict['loss_val_total'] = 0.5 * re_dict['loss_val_recon_x'] + 0.5 * re_dict['loss_val_cross_body']
        autoencoder.train()
        return re_dict


def temporal_pairwise_cosine_similarity(seqs_i, seqs_j):
    seq_len = seqs_i.size(2)
    seqs_i_exp = seqs_i.unsqueeze(3).repeat(1, 1, 1, seq_len)
    seqs_j_exp = seqs_j.unsqueeze(2).repeat(1, 1, seq_len, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=1)


def triplet_margin_loss(seqs_a, seqs_b, neg_range=(0.0, 0.5), margin=0.2):
    neg_start, neg_end = neg_range
    batch_size, _, seq_len = seqs_a.size()
    n_neg_all = seq_len ** 2
    n_neg = int(round(neg_end * n_neg_all))
    n_neg_discard = int(round(neg_start * n_neg_all))
    batch_size, _, seq_len = seqs_a.size()
    sim_aa = temporal_pairwise_cosine_similarity(seqs_a, seqs_a)
    sim_bb = temporal_pairwise_cosine_similarity(seqs_b, seqs_a)
    sim_ab = temporal_pairwise_cosine_similarity(seqs_a, seqs_b)
    sim_ba = sim_ab.transpose(1, 2)
    diff_ab = (sim_ab - sim_aa).reshape(batch_size, -1)
    diff_ba = (sim_ba - sim_bb).reshape(batch_size, -1)
    diff = torch.cat([diff_ab, diff_ba], dim=0)
    diff, _ = diff.topk(n_neg, dim=-1, sorted=True)
    diff = diff[:, n_neg_discard:]
    loss = diff + margin
    loss = loss.clamp(min=0.0)
    loss = loss.mean()
    return loss


class TransmomoTrainer(BaseTrainer):

    def __init__(self, config):
        super(TransmomoTrainer, self).__init__(config)
        self.angle_unit = np.pi / (config.K + 1)
        view_angles = np.array([(i * self.angle_unit) for i in range(1, config.K + 1)])
        x_angles = view_angles if config.rotation_axes[0] else np.array([0])
        z_angles = view_angles if config.rotation_axes[1] else np.array([0])
        y_angles = view_angles if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.angles = torch.tensor(angles).float()
        self.rotation_axes = torch.tensor(config.rotation_axes).float()
        self.rotation_axes_mask = [(_ > 0) for _ in config.rotation_axes]

    def dis_update(self, data, config):
        x_a = data['x'].detach()
        x_s = data['x_s'].detach()
        self.dis_opt.zero_grad()
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)
        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)
        inds = random.sample(list(range(self.angles.size(0))), config.K)
        angles = self.angles[inds].clone().detach()
        angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
        angles = angles.unsqueeze(0).unsqueeze(2)
        X_a_recon = self.autoencoder.decode(motion_a, body_a, view_a)
        x_a_trans = rotate_and_maybe_project(X_a_recon, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)
        x_a_exp = x_a.repeat_interleave(config.K, dim=0)
        self.loss_dis_trans = self.discriminator.calc_dis_loss(x_a_trans.detach(), x_a_exp)
        if config.trans_gan_ls_w > 0:
            X_s_recon = self.autoencoder.decode(motion_s, body_s, view_s)
            x_s_trans = rotate_and_maybe_project(X_s_recon, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)
            x_s_exp = x_s.repeat_interleave(config.K, dim=0)
            self.loss_dis_trans_ls = self.discriminator.calc_dis_loss(x_s_trans.detach(), x_s_exp)
        else:
            self.loss_dis_trans_ls = 0
        self.loss_dis_total = config.trans_gan_w * self.loss_dis_trans + config.trans_gan_ls_w * self.loss_dis_trans_ls
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def ae_update(self, data, config):
        x_a = data['x'].detach()
        x_s = data['x_s'].detach()
        self.ae_opt.zero_grad()
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)
        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)
        self.loss_inv_v_ls = self.recon_criterion(view_a, view_s) if config.inv_v_ls_w > 0 else 0
        self.loss_inv_m_ls = self.recon_criterion(motion_a, motion_s) if config.inv_m_ls_w > 0 else 0
        if config.triplet_b_w > 0:
            self.loss_triplet_b = triplet_margin_loss(body_a_seq, body_s_seq, neg_range=config.triplet_neg_range, margin=config.triplet_margin)
        else:
            self.loss_triplet_b = 0
        X_a_recon = self.autoencoder.decode(motion_a, body_a, view_a)
        x_a_recon = rotate_and_maybe_project(X_a_recon, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)
        X_s_recon = self.autoencoder.decode(motion_s, body_s, view_s)
        x_s_recon = rotate_and_maybe_project(X_s_recon, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)
        self.loss_recon_x = 0.5 * self.recon_criterion(x_a_recon, x_a) + 0.5 * self.recon_criterion(x_s_recon, x_s)
        X_as_recon = self.autoencoder.decode(motion_a, body_s, view_s)
        x_as_recon = rotate_and_maybe_project(X_as_recon, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)
        X_sa_recon = self.autoencoder.decode(motion_s, body_a, view_a)
        x_sa_recon = rotate_and_maybe_project(X_sa_recon, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)
        self.loss_cross_x = 0.5 * self.recon_criterion(x_as_recon, x_s) + 0.5 * self.recon_criterion(x_sa_recon, x_a)
        inds = random.sample(list(range(self.angles.size(0))), config.K)
        angles = self.angles[inds].clone().detach()
        angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
        angles = angles.unsqueeze(0).unsqueeze(2)
        x_a_trans = rotate_and_maybe_project(X_a_recon, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)
        x_s_trans = rotate_and_maybe_project(X_s_recon, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)
        self.loss_gan_trans = self.discriminator.calc_gen_loss(x_a_trans)
        self.loss_gan_trans_ls = self.discriminator.calc_gen_loss(x_s_trans) if config.trans_gan_ls_w > 0 else 0
        motion_a_trans = self.autoencoder.encode_motion(x_a_trans)
        body_a_trans, _ = self.autoencoder.encode_body(x_a_trans)
        view_a_trans, view_a_trans_seq = self.autoencoder.encode_view(x_a_trans)
        motion_s_trans = self.autoencoder.encode_motion(x_s_trans)
        body_s_trans, _ = self.autoencoder.encode_body(x_s_trans)
        self.loss_inv_m_trans = 0.5 * self.recon_criterion(motion_a_trans, motion_a.repeat_interleave(config.K, dim=0)) + 0.5 * self.recon_criterion(motion_s_trans, motion_s.repeat_interleave(config.K, dim=0))
        self.loss_inv_b_trans = 0.5 * self.recon_criterion(body_a_trans, body_a.repeat_interleave(config.K, dim=0)) + 0.5 * self.recon_criterion(body_s_trans, body_s.repeat_interleave(config.K, dim=0))
        if config.triplet_v_w > 0:
            view_a_seq_exp = view_a_seq.repeat_interleave(config.K, dim=0)
            self.loss_triplet_v = triplet_margin_loss(view_a_seq_exp, view_a_trans_seq, neg_range=config.triplet_neg_range, margin=config.triplet_margin)
        else:
            self.loss_triplet_v = 0
        self.loss_total = torch.tensor(0.0).float()
        self.loss_total += config.recon_x_w * self.loss_recon_x
        self.loss_total += config.cross_x_w * self.loss_cross_x
        self.loss_total += config.inv_v_ls_w * self.loss_inv_v_ls
        self.loss_total += config.inv_m_ls_w * self.loss_inv_m_ls
        self.loss_total += config.inv_b_trans_w * self.loss_inv_b_trans
        self.loss_total += config.inv_m_trans_w * self.loss_inv_m_trans
        self.loss_total += config.trans_gan_w * self.loss_gan_trans
        self.loss_total += config.trans_gan_ls_w * self.loss_gan_trans_ls
        self.loss_total += config.triplet_b_w * self.loss_triplet_b
        self.loss_total += config.triplet_v_w * self.loss_triplet_v
        self.loss_total.backward()
        self.ae_opt.step()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvDecoder,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvEncoder,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_yzhq97_transmomo_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

