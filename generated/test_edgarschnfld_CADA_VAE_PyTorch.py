import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
final_classifier = _module
models = _module
single_experiment = _module
vaemodel = _module

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


import scipy.io as sio


import torch


from sklearn import preprocessing


import copy


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


from sklearn.preprocessing import MinMaxScaler


from torch.nn import functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torch.autograd as autograd


from torch.utils import data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder_template(nn.Module):

    def __init__(self, input_dim, latent_size, hidden_size_rule, device):
        super(encoder_template, self).__init__()
        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]
        modules = []
        for i in range(len(self.layer_sizes) - 2):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(nn.ReLU())
        self.feature_encoder = nn.Sequential(*modules)
        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self.apply(weights_init)
        self

    def forward(self, x):
        h = self.feature_encoder(x)
        mu = self._mu(h)
        logvar = self._logvar(h)
        return mu, logvar


class decoder_template(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size_rule, device):
        super(decoder_template, self).__init__()
        self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(), nn.Linear(self.layer_sizes[1], output_dim))
        self.apply(weights_init)
        self

    def forward(self, x):
        return self.feature_decoder(x)


class LINEAR_LOGSOFTMAX(nn.Module):

    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class Model(nn.Module):

    def __init__(self, hyperparameters):
        super(Model, self).__init__()
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device)
        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10
        feature_dimensions = [2048, self.dataset.aux_data.size(1)]
        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            None
        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype], self.device)
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())
        self.optimizer = optim.Adam(parameters_to_optimize, lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)
        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i
        return mapped_label

    def trainstep(self, img, att):
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)
        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) + self.reconstruction_criterion(att_from_att, att)
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) + self.reconstruction_criterion(att_from_img, att)
        KLD = 0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp()) + 0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp())
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        distance = distance.sum()
        f1 = 1.0 * (self.current_epoch - self.warmup['cross_reconstruction']['start_epoch']) / (1.0 * (self.warmup['cross_reconstruction']['end_epoch'] - self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1 * (1.0 * self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.FloatTensor([min(max(f1, 0), self.warmup['cross_reconstruction']['factor'])])
        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])
        f3 = 1.0 * (self.current_epoch - self.warmup['distance']['start_epoch']) / (1.0 * (self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
        f3 = f3 * (1.0 * self.warmup['distance']['factor'])
        distance_factor = torch.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])
        self.optimizer.zero_grad()
        loss = reconstruction_loss - beta * KLD
        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor * cross_reconstruction_loss
        if distance_factor > 0:
            loss += distance_factor * distance
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_vae(self):
        losses = []
        self.dataloader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.dataset.novelclasses = self.dataset.novelclasses.long()
        self.dataset.seenclasses = self.dataset.seenclasses.long()
        self.train()
        self.reparameterize_with_noise = True
        None
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1
                label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                label = label.long()
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j]
                    data_from_modalities[j].requires_grad = False
                loss = self.trainstep(data_from_modalities[0], data_from_modalities[1])
                if i % 50 == 0:
                    None
                if i % 50 == 0 and i > 0:
                    losses.append(loss)
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        return losses

    def train_classifier(self, show_plots=False):
        if self.num_shots > 0:
            None
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')
        history = []
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses
        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']
        novelclass_aux_data = self.dataset.novelclass_aux_data
        seenclass_aux_data = self.dataset.seenclass_aux_data
        novel_corresponding_labels = self.dataset.novelclasses.long()
        seen_corresponding_labels = self.dataset.seenclasses.long()
        novel_test_feat = self.dataset.data['test_unseen']['resnet_features']
        seen_test_feat = self.dataset.data['test_seen']['resnet_features']
        test_seen_label = self.dataset.data['test_seen']['labels']
        test_novel_label = self.dataset.data['test_unseen']['labels']
        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']
        if self.generalized == False:
            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)
            if self.num_shots > 0:
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)
            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)
        if self.generalized:
            None
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            None
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)
        clf.apply(models.weights_init)
        with torch.no_grad():
            self.reparameterize_with_noise = False
            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).data
            test_novel_Y = test_novel_label
            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).data
            test_seen_Y = test_seen_label
            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)
                if sample_per_class != 0 and len(label) != 0:
                    classes = label.unique()
                    for i, s in enumerate(classes):
                        features_of_that_class = features[(label == s), :]
                        multiplier = torch.ceil(torch.FloatTensor([max(1, sample_per_class / features_of_that_class.size(0))])).long().item()
                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)
                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat((features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)), dim=0)
                    return features_to_return, labels_to_return
                else:
                    return torch.FloatTensor([]), torch.LongTensor([])
            img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(train_seen_feat, train_seen_label, self.img_seen_samples)
            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(train_unseen_feat, train_unseen_label, self.img_unseen_samples)
            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(novelclass_aux_data, novel_corresponding_labels, self.att_unseen_samples)
            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(seenclass_aux_data, seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.FloatTensor([])
            z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
            train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
            train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]
            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)
        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X, test_novel_Y, cls_seenclasses, cls_novelclasses, self.num_classes, self.device, self.lr_cls, 0.5, 1, self.classifier_batch_size, self.generalized)
        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()
            if self.generalized:
                None
                history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item()])
            else:
                None
                history.append([0, torch.tensor(cls.acc).item(), 0])
        if self.generalized:
            return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item(), history
        else:
            return 0, torch.tensor(cls.acc).item(), 0, history


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LINEAR_LOGSOFTMAX,
     lambda: ([], {'input_dim': 4, 'nclass': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (encoder_template,
     lambda: ([], {'input_dim': 4, 'latent_size': 4, 'hidden_size_rule': [4, 4], 'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_edgarschnfld_CADA_VAE_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

