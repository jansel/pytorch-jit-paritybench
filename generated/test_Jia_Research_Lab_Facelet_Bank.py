import sys
_module = sys.modules[__name__]
del sys
demo2_facelet = _module
data = _module
base_dataset = _module
testData = _module
trainDataset = _module
walk_data = _module
global_vars = _module
network = _module
base_network = _module
decoder = _module
facelet_net = _module
test_facelet_net = _module
train_facelet_net = _module
util = _module
framework = _module
opt = _module
sha256 = _module
test_parse = _module
train_parse = _module
util = _module

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


from collections import OrderedDict


import torch


import torch.nn as nn


from torch import nn


import torch.optim as optim


import functools


from torch.utils.data import DataLoader


import numpy as np


from torch.autograd import Variable


model_urls = {'older':
    'https://www.dropbox.com/s/8irg2hguatwdm6v/older.pth?dl=1', 'younger':
    'https://www.dropbox.com/s/drsx7slvmjdpwuq/younger.pth?dl=1',
    'facehair':
    'https://www.dropbox.com/s/xjd2xh53vw82ces/facehair.pth?dl=1',
    'masculinization':
    'https://www.dropbox.com/s/20q30jnn02qn7n0/masculinization.pth?dl=1',
    'feminization':
    'https://www.dropbox.com/s/5timkh7fuclwk9m/feminization.pth?dl=1',
    'vgg19g':
    'https://www.dropbox.com/s/4lbt58k10o84l5h/vgg19g-4aff041b.pth?dl=1',
    'vgg_decoder_res':
    'https://www.dropbox.com/s/t8vsobxz8avsmj0/decoder_res-9d1e0fe5.pth?dl=1'}


class BaseModel(object):

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self, x):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def print_current_errors(self, epoch, i, record_file=None):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        if record_file is not None:
            with open(record_file + '/loss.txt', 'w') as f:
                print(message, file=f)

    def save(self, label):
        pass

    def load(self, pretrain_path, label):
        pass

    def save_network(self, network, save_dir, label):
        save_filename = '%s.pth' % label
        save_path = os.path.join(save_dir, save_filename)
        print('saving %s in %s' % (save_filename, save_path))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    def resume_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        print('loading %s from %s' % (save_filename, save_path))
        network.load_state_dict(torch.load(save_path))

    def load_network(self, network, pretrain_path, label):
        filename = '%s.pth' % label
        save_path = os.path.join(pretrain_path, filename)
        print('loading %s from %s' % (filename, pretrain_path))
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class VGG(nn.Module, BaseModel):

    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.features_1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(
            3, 64, kernel_size=3, padding=1)), ('relu1_1', nn.ReLU(inplace=
            True)), ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)
            ), ('relu1_2', nn.ReLU(inplace=True)), ('pool1', nn.MaxPool2d(2,
            2)), ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)), ('conv2_2', nn.Conv2d(128, 
            128, kernel_size=3, padding=1)), ('relu2_2', nn.ReLU(inplace=
            True)), ('pool2', nn.MaxPool2d(2, 2)), ('conv3_1', nn.Conv2d(
            128, 256, kernel_size=3, padding=1)), ('relu3_1', nn.ReLU(
            inplace=True))]))
        self.features_2 = nn.Sequential(OrderedDict([('conv3_2', nn.Conv2d(
            256, 256, kernel_size=3, padding=1)), ('relu3_2', nn.ReLU(
            inplace=True)), ('conv3_3', nn.Conv2d(256, 256, kernel_size=3,
            padding=1)), ('relu3_3', nn.ReLU(inplace=True)), ('conv3_4', nn
            .Conv2d(256, 256, kernel_size=3, padding=1)), ('relu3_5', nn.
            ReLU(inplace=True)), ('pool3', nn.MaxPool2d(2, 2)), ('conv4_1',
            nn.Conv2d(256, 512, kernel_size=3, padding=1)), ('relu4_1', nn.
            ReLU(inplace=True))]))
        self.features_3 = nn.Sequential(OrderedDict([('conv4_2', nn.Conv2d(
            512, 512, kernel_size=3, padding=1)), ('relu4_2', nn.ReLU(
            inplace=True)), ('conv4_3', nn.Conv2d(512, 512, kernel_size=3,
            padding=1)), ('relu4_3', nn.ReLU(inplace=True)), ('conv4_4', nn
            .Conv2d(512, 512, kernel_size=3, padding=1)), ('relu4_4', nn.
            ReLU(inplace=True)), ('pool4', nn.MaxPool2d(2, 2)), ('conv5_1',
            nn.Conv2d(512, 512, kernel_size=3, padding=1)), ('relu5_1', nn.
            ReLU(inplace=True))]))
        if pretrained:
            None
            state_dict = torch.utils.model_zoo.load_url(model_urls['vgg19g'
                ], 'facelet_bank')
            model_dict = self.state_dict()
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3


class Vgg_recon(nn.Module):

    def __init__(self, drop_rate=0):
        super(Vgg_recon, self).__init__()
        self.recon5 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate)
        self.upool4 = _TransitionUp(512, 512)
        self.upsample4 = _Upsample(512, 512)
        self.recon4 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate)
        self.upool3 = _TransitionUp(512, 256)
        self.upsample3 = _Upsample(512, 256)
        self.recon3 = _PoolingBlock(3, 256, 256, drop_rate=drop_rate)
        self.upool2 = _TransitionUp(256, 128)
        self.upsample2 = _Upsample(256, 128)
        self.recon2 = _PoolingBlock(2, 128, 128, drop_rate=drop_rate)
        self.upool1 = _TransitionUp(128, 64)
        self.upsample1 = _Upsample(128, 64)
        self.recon1 = _PoolingBlock(1, 64, 64, drop_rate=drop_rate)
        self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, fy):
        features_1, features_2, features_3 = fy
        recon5 = self.recon5(features_3)
        recon5 = nn.functional.upsample(recon5, scale_factor=2, mode='bilinear'
            )
        upool4 = self.upsample4(recon5)
        recon4 = self.recon4(upool4 + features_2)
        recon4 = nn.functional.upsample(recon4, scale_factor=2, mode='bilinear'
            )
        upool3 = self.upsample3(recon4)
        recon3 = self.recon3(upool3 + features_1)
        recon3 = nn.functional.upsample(recon3, scale_factor=2, mode='bilinear'
            )
        upool2 = self.upsample2(recon3)
        recon2 = self.recon2(upool2)
        recon2 = nn.functional.upsample(recon2, scale_factor=2, mode='bilinear'
            )
        upool1 = self.upsample1(recon2)
        recon1 = self.recon1(upool1)
        recon0 = self.recon0(recon1)
        return recon0


class _PoolingBlock(nn.Sequential):

    def __init__(self, n_convs, n_input_filters, n_output_filters, drop_rate):
        super(_PoolingBlock, self).__init__()
        for i in range(n_convs):
            self.add_module('conv.%d' % (i + 1), nn.Conv2d(n_input_filters if
                i == 0 else n_output_filters, n_output_filters, kernel_size
                =3, padding=1))
            self.add_module('norm.%d' % (i + 1), nn.BatchNorm2d(
                n_output_filters))
            self.add_module('relu.%d' % (i + 1), nn.ReLU(inplace=True))
            if drop_rate > 0:
                self.add_module('drop.%d' % (i + 1), nn.Dropout(p=drop_rate))


class _TransitionUp(nn.Sequential):

    def __init__(self, n_input_filters, n_output_filters):
        super(_TransitionUp, self).__init__()
        self.add_module('unpool.conv', nn.ConvTranspose2d(n_input_filters,
            n_output_filters, kernel_size=4, stride=2, padding=1))
        self.add_module('unpool.norm', nn.BatchNorm2d(n_output_filters))


class _Upsample(nn.Sequential):

    def __init__(self, n_input_filters, n_output_filters):
        super(_Upsample, self).__init__()
        self.add_module('interp.conv', nn.Conv2d(n_input_filters,
            n_output_filters, kernel_size=3, padding=1))
        self.add_module('interp.norm', nn.BatchNorm2d(n_output_filters))


class vgg_decoder(base_network.BaseModel, nn.Module):

    def __init__(self, pretrained=True):
        super(vgg_decoder, self).__init__()
        self._define_model()
        if pretrained:
            None
            state_dict = torch.utils.model_zoo.load_url(model_urls[
                'vgg_decoder_res'], model_dir='facelet_bank')
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)

    def _define_model(self):
        self.model = base_network.Vgg_recon()

    def forward(self, fy, img=None):
        fy = [util.toVariable(f) for f in fy]
        y = self.model.forward(fy)
        y = y + img
        return y

    def load(self, pretrain_path, epoch_label='latest'):
        self.load_network(pretrain_path, self.model, 'recon', epoch_label)


facelet_path = 'facelet_bank'


class opt(object):

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def merge_dict(self, d):
        self.__dict__.update(d)
        return self

    def merge_opt(self, o):
        d = vars(o)
        self.__dict__.update(d)
        return self


class simpleCNNGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=None, n_blocks=3):
        assert n_blocks >= 0
        super(simpleCNNGenerator, self).__init__()
        ngf = input_nc
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer is not None:
            model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf,
                kernel_size=3, padding=0, bias=use_bias), norm_layer(ngf),
                nn.ReLU(inplace=True)]
        else:
            model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf,
                kernel_size=3, padding=0), nn.ReLU(inplace=True)]
        for i in range(n_blocks - 2):
            if norm_layer is not None:
                model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1,
                    bias=use_bias), norm_layer(ngf), nn.ReLU(inplace=True)]
            else:
                model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1), nn
                    .ReLU(inplace=True)]
        if norm_layer is not None:
            model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, output_nc,
                kernel_size=3, padding=0, bias=use_bias)]
        else:
            model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, output_nc,
                kernel_size=3, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Jia_Research_Lab_Facelet_Bank(_paritybench_base):
    pass
    def test_000(self):
        self._check(simpleCNNGenerator(*[], **{'input_nc': 4, 'output_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

