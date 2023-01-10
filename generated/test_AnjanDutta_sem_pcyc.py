import sys
_module = sys.modules[__name__]
del sys
data = _module
download_gdrive = _module
itq = _module
logger = _module
losses = _module
models = _module
options = _module
test = _module
train = _module
utils = _module
train_image = _module
train_sketch = _module

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


import torch.utils.data as data


import torch


from numpy.linalg import multi_dot


from sklearn.feature_selection import SelectKBest


from sklearn.feature_selection import chi2


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


from torchvision import models


import torch.nn.functional as F


import time


from sklearn.neighbors import KDTree


from scipy.spatial.distance import cdist


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


from torch.utils.data.sampler import WeightedRandomSampler


import random


import itertools


from sklearn.metrics import average_precision_score


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = 0.7 + 0.3 * torch.rand(input.size(0))
        else:
            target_tensor = 0.3 * torch.rand(input.size(0))
        if input.is_cuda:
            target_tensor = target_tensor
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input.squeeze(), target_tensor)


class VGGNetFeats(nn.Module):

    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(4096, 512))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class GaussianNoiseLayer(nn.Module):

    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise
            x = x + noise
        return x


class Generator(nn.Module):

    def __init__(self, in_dim=512, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator, self).__init__()
        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        self.gen = nn.Sequential(*modules)

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):

    def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False, use_sigmoid=False):
        super(Discriminator, self).__init__()
        hid_dim = int(in_dim / 2)
        modules = list()
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        if use_sigmoid:
            modules.append(nn.Sigmoid())
        self.disc = nn.Sequential(*modules)

    def forward(self, x):
        return self.disc(x)


class AutoEncoder(nn.Module):

    def __init__(self, dim=300, hid_dim=300, nlayer=1):
        super(AutoEncoder, self).__init__()
        steps_down = np.linspace(dim, hid_dim, num=nlayer + 1, dtype=np.int).tolist()
        modules = []
        for i in range(nlayer):
            modules.append(nn.Linear(steps_down[i], steps_down[i + 1]))
            modules.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*modules)
        steps_up = np.linspace(hid_dim, dim, num=nlayer + 1, dtype=np.int).tolist()
        modules = []
        for i in range(nlayer):
            modules.append(nn.Linear(steps_up[i], steps_up[i + 1]))
            modules.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*modules)

    def forward(self, x):
        xenc = self.enc(x)
        xrec = self.dec(xenc)
        return xenc, xrec


class SEM_PCYC(nn.Module):

    def __init__(self, params_model):
        super(SEM_PCYC, self).__init__()
        None
        self.dim_out = params_model['dim_out']
        self.sem_dim = params_model['sem_dim']
        self.num_clss = params_model['num_clss']
        self.sketch_model = VGGNetFeats(pretrained=False, finetune=False)
        self.load_weight(self.sketch_model, params_model['path_sketch_model'], 'sketch')
        self.image_model = VGGNetFeats(pretrained=False, finetune=False)
        self.load_weight(self.image_model, params_model['path_image_model'], 'image')
        self.sem = []
        for f in params_model['files_semantic_labels']:
            self.sem.append(np.load(f, allow_pickle=True).item())
        self.dict_clss = params_model['dict_clss']
        None
        None
        self.gen_sk2se = Generator(in_dim=512, out_dim=self.dim_out, noise=False, use_dropout=True)
        self.gen_im2se = Generator(in_dim=512, out_dim=self.dim_out, noise=False, use_dropout=True)
        self.gen_se2sk = Generator(in_dim=self.dim_out, out_dim=512, noise=False, use_dropout=True)
        self.gen_se2im = Generator(in_dim=self.dim_out, out_dim=512, noise=False, use_dropout=True)
        self.disc_se = Discriminator(in_dim=self.dim_out, noise=True, use_batchnorm=True)
        self.disc_sk = Discriminator(in_dim=512, noise=True, use_batchnorm=True)
        self.disc_im = Discriminator(in_dim=512, noise=True, use_batchnorm=True)
        self.aut_enc = AutoEncoder(dim=self.sem_dim, hid_dim=self.dim_out, nlayer=1)
        self.classifier_sk = nn.Linear(512, self.num_clss, bias=False)
        self.classifier_im = nn.Linear(512, self.num_clss, bias=False)
        self.classifier_se = nn.Linear(self.dim_out, self.num_clss, bias=False)
        for param in self.classifier_sk.parameters():
            param.requires_grad = False
        for param in self.classifier_im.parameters():
            param.requires_grad = False
        for param in self.classifier_se.parameters():
            param.requires_grad = False
        None
        None
        self.lr = params_model['lr']
        self.gamma = params_model['gamma']
        self.momentum = params_model['momentum']
        self.milestones = params_model['milestones']
        self.optimizer_gen = optim.Adam(list(self.gen_sk2se.parameters()) + list(self.gen_im2se.parameters()) + list(self.gen_se2sk.parameters()) + list(self.gen_se2im.parameters()), lr=self.lr)
        self.optimizer_disc = optim.SGD(list(self.disc_se.parameters()) + list(self.disc_sk.parameters()) + list(self.disc_im.parameters()), lr=self.lr, momentum=self.momentum)
        self.optimizer_ae = optim.SGD(self.aut_enc.parameters(), lr=100 * self.lr, momentum=self.momentum)
        self.scheduler_gen = optim.lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=self.milestones, gamma=self.gamma)
        self.scheduler_disc = optim.lr_scheduler.MultiStepLR(self.optimizer_disc, milestones=self.milestones, gamma=self.gamma)
        self.scheduler_ae = optim.lr_scheduler.MultiStepLR(self.optimizer_ae, milestones=self.milestones, gamma=self.gamma)
        None
        None
        self.lambda_se = params_model['lambda_se']
        self.lambda_im = params_model['lambda_im']
        self.lambda_sk = params_model['lambda_sk']
        self.lambda_gen_cyc = params_model['lambda_gen_cyc']
        self.lambda_gen_adv = params_model['lambda_gen_adv']
        self.lambda_gen_cls = params_model['lambda_gen_cls']
        self.lambda_gen_reg = params_model['lambda_gen_reg']
        self.lambda_disc_se = params_model['lambda_disc_se']
        self.lambda_disc_sk = params_model['lambda_disc_sk']
        self.lambda_disc_im = params_model['lambda_disc_im']
        self.lambda_regular = params_model['lambda_regular']
        self.criterion_gan = GANLoss(use_lsgan=True)
        self.criterion_cyc = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        None
        None
        self.sk_fe = torch.zeros(1)
        self.sk_em = torch.zeros(1)
        self.im_fe = torch.zeros(1)
        self.im_em = torch.zeros(1)
        self.se_em_enc = torch.zeros(1)
        self.se_em_rec = torch.zeros(1)
        self.im2se_em = torch.zeros(1)
        self.sk2se_em = torch.zeros(1)
        self.se2im_em = torch.zeros(1)
        self.se2sk_em = torch.zeros(1)
        self.im_em_hat = torch.zeros(1)
        self.sk_em_hat = torch.zeros(1)
        self.se_em_hat1 = torch.zeros(1)
        self.se_em_hat2 = torch.zeros(1)
        None

    def load_weight(self, model, path, type='sketch'):
        checkpoint = torch.load(os.path.join(path, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict_' + type])

    def forward(self, sk, im, se):
        self.sk_fe = self.sketch_model(sk)
        self.im_fe = self.image_model(im)
        self.se_em_enc, self.se_em_rec = self.aut_enc(se)
        self.im2se_em = self.gen_im2se(self.im_fe)
        self.sk2se_em = self.gen_sk2se(self.sk_fe)
        self.se2im_em = self.gen_se2im(self.se_em_enc.detach())
        self.se2sk_em = self.gen_se2sk(self.se_em_enc.detach())
        self.im_em_hat = self.gen_se2im(self.im2se_em)
        self.sk_em_hat = self.gen_se2sk(self.sk2se_em)
        self.se_em_hat1 = self.gen_sk2se(self.se2sk_em)
        self.se_em_hat2 = self.gen_im2se(self.se2im_em)

    def backward(self, se, cl):
        regularizer = 0
        if self.lambda_regular > 0:
            for l in range(len(self.aut_enc.enc)):
                if self.aut_enc.enc[l]._get_name() == 'Linear':
                    regularizer = regularizer + torch.norm(self.aut_enc.enc[l].weight, 2, dim=0).sum()
        loss_aut_enc = self.criterion_reg(self.se_em_rec, se) + self.lambda_regular * regularizer / cl.shape[0]
        self.optimizer_ae.zero_grad()
        loss_aut_enc.backward(retain_graph=True)
        self.optimizer_ae.step()
        loss_gen_adv = self.criterion_gan(self.disc_se(self.im2se_em), True) + self.criterion_gan(self.disc_se(self.sk2se_em), True) + self.criterion_gan(self.disc_im(self.se2im_em), True) + self.criterion_gan(self.disc_sk(self.se2sk_em), True)
        loss_gen_adv = self.lambda_gen_adv * loss_gen_adv
        loss_gen_cyc = self.lambda_im * self.criterion_cyc(self.im_em_hat, self.im_fe) + self.lambda_sk * self.criterion_cyc(self.sk_em_hat, self.sk_fe) + self.lambda_se * (self.criterion_cyc(self.se_em_hat1, self.se_em_enc.detach()) + self.criterion_cyc(self.se_em_hat2, self.se_em_enc.detach()))
        loss_gen_cyc = self.lambda_gen_cyc * loss_gen_cyc
        loss_gen_cls = self.lambda_se * (self.criterion_cls(self.classifier_se(self.im2se_em), cl) + self.criterion_cls(self.classifier_se(self.sk2se_em), cl)) + self.lambda_sk * self.criterion_cls(self.classifier_sk(self.se2sk_em), cl) + self.lambda_im * self.criterion_cls(self.classifier_im(self.se2im_em), cl)
        loss_gen_cls = self.lambda_gen_cls * loss_gen_cls
        loss_gen_reg = self.lambda_se * (self.criterion_reg(self.im2se_em, self.sk2se_em) + self.criterion_reg(self.sk2se_em, self.im2se_em)) + self.lambda_im * self.criterion_reg(self.se2im_em, self.im_fe) + self.lambda_sk * self.criterion_reg(self.se2sk_em, self.sk_fe)
        loss_gen_reg = self.lambda_gen_reg * loss_gen_reg
        loss_gen = loss_gen_adv + loss_gen_cyc + loss_gen_cls + loss_gen_reg
        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        self.optimizer_gen.step()
        self.optimizer_disc.zero_grad()
        loss_disc_se = 2 * self.criterion_gan(self.disc_se(self.se_em_enc), True) + self.criterion_gan(self.disc_se(self.im2se_em), False) + self.criterion_gan(self.disc_se(self.sk2se_em), False)
        loss_disc_se = self.lambda_disc_se * loss_disc_se
        loss_disc_se.backward(retain_graph=True)
        loss_disc_sk = self.criterion_gan(self.disc_sk(self.sk_fe), True) + self.criterion_gan(self.disc_sk(self.se2sk_em), False)
        loss_disc_sk = self.lambda_disc_sk * loss_disc_sk
        loss_disc_sk.backward(retain_graph=True)
        loss_disc_im = self.criterion_gan(self.disc_im(self.im_fe), True) + self.criterion_gan(self.disc_im(self.se2im_em), False)
        loss_disc_im = self.lambda_disc_im * loss_disc_im
        loss_disc_im.backward()
        self.optimizer_disc.step()
        loss_disc = loss_disc_se + loss_disc_sk + loss_disc_im
        loss = {'aut_enc': loss_aut_enc, 'gen_adv': loss_gen_adv, 'gen_cyc': loss_gen_cyc, 'gen_cls': loss_gen_cls, 'gen_reg': loss_gen_reg, 'gen': loss_gen, 'disc_se': loss_disc_se, 'disc_sk': loss_disc_sk, 'disc_im': loss_disc_im, 'disc': loss_disc}
        return loss

    def optimize_params(self, sk, im, cl):
        num_cls = torch.from_numpy(utils.numeric_classes(cl, self.dict_clss))
        se = np.zeros((len(cl), self.sem_dim), dtype=np.float32)
        for i, c in enumerate(cl):
            se_c = np.array([], dtype=np.float32)
            for s in self.sem:
                se_c = np.concatenate((se_c, s.get(c).astype(np.float32)), axis=0)
            se[i] = se_c
        se = torch.from_numpy(se)
        if torch.cuda.is_available:
            se = se
        self.forward(sk, im, se)
        loss = self.backward(se, num_cls)
        return loss

    def get_sketch_embeddings(self, sk):
        sk_em = self.gen_sk2se(self.sketch_model(sk))
        return sk_em

    def get_image_embeddings(self, im):
        im_em = self.gen_im2se(self.image_model(im))
        return im_em


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (GaussianNoiseLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_AnjanDutta_sem_pcyc(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

