import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
aligned_pair_dataset = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
dataset = _module
enhance = _module
main = _module
model = _module
trainer = _module
configs = _module
perceptual_loss = _module
show_skeleton_on_RGB = _module
spectral_norm = _module
models = _module
base_model = _module
models = _module
networks = _module
networks_modified = _module
pix2pixHD_model = _module
pose2vidHD_model = _module
spynet = _module
ui_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
pose_utils = _module
test = _module
test_video = _module
train = _module
util = _module
html = _module
image_pool = _module
util = _module
visualizer = _module

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


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


import random


import torch.utils.data


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.backends import cudnn


import torch.nn as nn


import torch.nn.parallel


from torch.nn import functional


import time


from torch import nn


import torch.optim as optim


from torchvision.models import vgg


from torch.nn.functional import normalize


from torch.nn.parameter import Parameter


import functools


from torch.autograd import Variable


from torchvision import models


import math


from torch.nn.functional import upsample


from collections import OrderedDict


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class VGG_perceptual_loss(nn.Module):

    def __init__(self, pretrained=True, device='cuda'):
        super(VGG_perceptual_loss, self).__init__()
        self.device = device
        self.loss_function = nn.L1Loss()
        self.vgg_features = vgg.make_layers(vgg.cfg['D'])
        if pretrained:
            self.vgg_features.load_state_dict(torch.load('utils/vgg16_pretrained_features.pth'))
        self.vgg_features
        for params in self.vgg_features.parameters():
            params.requires_grad = False
        self.layer_name_mapping = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}

    def forward(self, input, target):
        loss = torch.tensor(0.0)
        for name, module in self.vgg_features._modules.items():
            input = module(input)
            target = module(target)
            if name in self.layer_name_mapping:
                loss += self.loss_function(input, target)
        return 0.1 * loss


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
            if network_label == 'G':
                raise 'Generator must exist!'
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        None
                except:
                    None
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        not_initialized = Set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    None
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class LocalEnhancer(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        ngf_global = ngf * 2 ** n_local_enhancers
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.Sequential(*model_global)
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True), nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        output_prev = self.model(input_downsampled[-1])
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, (0)] + b, indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, (0)] + b, indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


arguments_strModel = 'sintel-final'


class SpyNetwork(torch.nn.Module):

    def __init__(self, optical_flow_model='F'):
        super(SpyNetwork, self).__init__()


        class Preprocess(torch.nn.Module):

            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tensorInput):
                tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229
                return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


        class Basic(torch.nn.Module):

            def __init__(self, intLevel):
                super(Basic, self).__init__()
                self.moduleBasic = torch.nn.Sequential(torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))
                if intLevel == 5:
                    if arguments_strModel == '3' or arguments_strModel == '4':
                        intLevel = 4
                for intConv in range(5):
                    self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel + '-' + str(intConv + 1) + '-weight.t7'))
                    self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua('./models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel + '-' + str(intConv + 1) + '-bias.t7'))

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)


        class Backward(torch.nn.Module):

            def __init__(self):
                super(Backward, self).__init__()

            def forward(self, tensorInput, tensorFlow):
                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
                    self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1)
                tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
                return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        self.modulePreprocess = Preprocess()
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
        self.moduleBackward = Backward()

    def estimate(self, tensorFirst, tensorSecond):
        tensorFlow = []
        tensorFirst = [self.modulePreprocess(tensorFirst)]
        tensorSecond = [self.modulePreprocess(tensorSecond)]
        for intLevel in range(5):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)))
        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.upsample(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=(0, 0, 0, 1), mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=(0, 1, 0, 0), mode='replicate')
            tensorFlow = self.moduleBasic[intLevel](torch.cat((tensorFirst[intLevel], self.moduleBackward(tensorSecond[intLevel], tensorUpsampled), tensorUpsampled), 1)) + tensorUpsampled
        return tensorFlow

    def forward(self, tensorFirst, tensorSecond):
        assert tensorFirst.size(2) == tensorSecond.size(2)
        assert tensorFirst.size(3) == tensorSecond.size(3)
        intWidth = tensorFirst.size(3)
        intHeight = tensorFirst.size(2)
        if intWidth % 32 == 0 and intHeight % 32 == 0:
            tensorFlow = self.estimate(tensorFirst, tensorSecond)
        else:
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))
            tensorPreprocessedFirst = torch.nn.functional.upsample(input=tensorFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tensorPreprocessedSecond = torch.nn.functional.upsample(input=tensorSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tensorFlow = self.estimate(tensorPreprocessedFirst, tensorPreprocessedSecond)
            tensorFlow = torch.nn.functional.upsample(input=tensorFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
            tensorFlow[:, (0), :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tensorFlow[:, (1), :, :] *= float(intHeight) / float(intPreprocessedHeight)
        return tensorFlow[0]


class FlowLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(FlowLoss, self).__init__()
        self.flownet = SpyNetwork()
        self.criterion = nn.L1Loss()
        self.count = 0

    def forward(self, real1, real2, fake1, fake2):
        real_flow, fake_flow = self.flownet(real1, real2), self.flownet(fake1, fake2)
        return self.criterion(real_flow, fake_flow)


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = True, use_gan_feat_loss, use_vgg_loss, True, True

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for l, f in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            None
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
            if opt.niter_fix_global > 0:
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    finetune_list = Set()
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                None
                None
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        if real_image is not None:
            real_image = Variable(real_image.data)
        if self.use_features:
            if self.opt.load_features:
                feat_map = Variable(feat_map.data)
        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), None if not infer else fake_image]

    def inference(self, label, inst):
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)
        if self.use_features:
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[(num // 2), :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            None
        self.old_lr = lr


class Pose2VidHDModel(BaseModel):

    def name(self):
        return 'Pose2VidHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_flow_loss):
        flags = True, use_gan_feat_loss, use_vgg_loss, use_flow_loss, True, True

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_flow, d_real, d_fake):
            return [l for l, f in zip((g_gan, g_gan_feat, g_vgg, g_flow, d_real, d_fake), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        netG_input_nc += opt.output_nc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            netD_input_nc *= 2
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            None
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_flow_loss)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            if not opt.no_flow_loss:
                self.nelem = 288 * 512 if opt.dataroot.find('512') != -1 else 576 * 1024
                self.criterionFlow = networks.FlowLoss(self.gpu_ids)
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_flow', 'D_real', 'D_fake')
            if opt.niter_fix_global > 0:
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    finetune_list = Set()
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                None
                None
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        if real_image is not None:
            real_image = Variable(real_image.data)
        if self.use_features:
            if self.opt.load_features:
                feat_map = Variable(feat_map.data)
        return input_label, inst_map, real_image, feat_map

    def forward(self, label, real_image):
        label = label.data
        real_image = real_image.data
        gt1 = real_image[:, (0), (...)]
        gt2 = real_image[:, (1), (...)]
        x1 = label[:, (0), (...)]
        x2 = label[:, (1), (...)]
        zero = torch.zeros_like(gt1)
        y1 = self.netG.forward(torch.cat((x1, zero), 1))
        y2 = self.netG.forward(torch.cat((x2, y1), 1))
        pred_fake_pool = self.netD.forward(torch.cat((x1, x2, y1.detach(), y2.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.netD.forward(torch.cat((x1, x2, gt1.detach(), gt2.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD.forward(torch.cat((x1, x2, y1, y2), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = (self.criterionVGG(y1, gt1) + self.criterionVGG(y2, gt2)) * self.opt.lambda_feat
        loss_G_flow = 0
        if not self.opt.no_flow_loss:
            loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow / self.nelem
        y1_clean = torch.squeeze(y1.detach())
        y2_clean = torch.squeeze(y2.detach())
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_flow, loss_D_real, loss_D_fake), torch.cat((y1_clean, y2_clean), dim=2)]

    def train_one_step(self, data, save_fake=False):
        losses, generated = self.forward(Variable(data['label']), Variable(data['image']))
        losses = [(torch.mean(x) if not isinstance(x, int) else x) for x in losses]
        loss_dict = dict(zip(self.loss_names, losses))
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_flow', 0)
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        return loss_dict, generated if save_fake else None

    def inference(self, label, inst, prev_frame):
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)
        prev_frame = Variable(prev_frame.data)
        if self.use_features:
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        input_concat = torch.cat((input_concat, prev_frame), dim=1)
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[(num // 2), :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            None
        self.old_lr = lr


class InferenceModel(Pose2VidHDModel):

    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


class UIModel(BaseModel):

    def name(self):
        return 'UIModel'

    def initialize(self, opt):
        assert not opt.isTrain
        BaseModel.initialize(self, opt)
        self.use_features = opt.instance_feat or opt.label_feat
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.load_network(self.netG, 'G', opt.which_epoch)
        None

    def toTensor(self, img, normalize=False):
        tensor = torch.from_numpy(np.array(img, np.int32, copy=False))
        tensor = tensor.view(1, img.size[1], img.size[0], len(img.mode))
        tensor = tensor.transpose(1, 2).transpose(1, 3).contiguous()
        if normalize:
            return (tensor.float() / 255.0 - 0.5) / 0.5
        return tensor.float()

    def load_image(self, label_path, inst_path, feat_path):
        opt = self.opt
        label_img = Image.open(label_path)
        if label_path.find('face') != -1:
            label_img = label_img.convert('L')
        ow, oh = label_img.size
        w = opt.loadSize
        h = int(w * oh / ow)
        label_img = label_img.resize((w, h), Image.NEAREST)
        label_map = self.toTensor(label_img)
        self.label_map = label_map
        oneHot_size = 1, opt.label_nc, h, w
        input_label = self.Tensor(torch.Size(oneHot_size)).zero_()
        self.input_label = input_label.scatter_(1, label_map.long(), 1.0)
        if not opt.no_instance:
            inst_img = Image.open(inst_path)
            inst_img = inst_img.resize((w, h), Image.NEAREST)
            self.inst_map = self.toTensor(inst_img)
            self.edge_map = self.get_edges(self.inst_map)
            self.net_input = Variable(torch.cat((self.input_label, self.edge_map), dim=1), volatile=True)
        else:
            self.net_input = Variable(self.input_label, volatile=True)
        self.features_clustered = np.load(feat_path).item()
        self.object_map = self.inst_map if opt.instance_feat else self.label_map
        object_np = self.object_map.cpu().numpy().astype(int)
        self.feat_map = self.Tensor(1, opt.feat_num, h, w).zero_()
        self.cluster_indices = np.zeros(self.opt.label_nc, np.uint8)
        for i in np.unique(object_np):
            label = i if i < 1000 else i // 1000
            if label in self.features_clustered:
                feat = self.features_clustered[label]
                np.random.seed(i + 1)
                cluster_idx = np.random.randint(0, feat.shape[0])
                self.cluster_indices[label] = cluster_idx
                idx = (self.object_map == i).nonzero()
                self.set_features(idx, feat, cluster_idx)
        self.net_input_original = self.net_input.clone()
        self.label_map_original = self.label_map.clone()
        self.feat_map_original = self.feat_map.clone()
        if not opt.no_instance:
            self.inst_map_original = self.inst_map.clone()

    def reset(self):
        self.net_input = self.net_input_prev = self.net_input_original.clone()
        self.label_map = self.label_map_prev = self.label_map_original.clone()
        self.feat_map = self.feat_map_prev = self.feat_map_original.clone()
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev = self.inst_map_original.clone()
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def undo(self):
        self.net_input = self.net_input_prev
        self.label_map = self.label_map_prev
        self.feat_map = self.feat_map_prev
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def change_labels(self, click_src, click_tgt):
        y_src, x_src = click_src[0], click_src[1]
        y_tgt, x_tgt = click_tgt[0], click_tgt[1]
        label_src = int(self.label_map[0, 0, y_src, x_src])
        inst_src = self.inst_map[0, 0, y_src, x_src]
        label_tgt = int(self.label_map[0, 0, y_tgt, x_tgt])
        inst_tgt = self.inst_map[0, 0, y_tgt, x_tgt]
        idx_src = (self.inst_map == inst_src).nonzero()
        if idx_src.shape:
            self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_src, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            if inst_tgt > 1000:
                tgt_indices = (self.inst_map > label_tgt * 1000) & (self.inst_map < (label_tgt + 1) * 1000)
                inst_tgt = self.inst_map[tgt_indices].max() + 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = inst_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            idx_tgt = (self.inst_map == inst_tgt).nonzero()
            if idx_tgt.shape:
                self.copy_features(idx_src, idx_tgt[(0), :])
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_strokes(self, click_src, label_tgt, bw, save):
        size = self.net_input.size()
        h, w = size[2], size[3]
        idx_src = torch.LongTensor(bw ** 2, 4).fill_(0)
        for i in range(bw):
            idx_src[i * bw:(i + 1) * bw, (2)] = min(h - 1, max(0, click_src[0] - bw // 2 + i))
            for j in range(bw):
                idx_src[i * bw + j, 3] = min(w - 1, max(0, click_src[1] - bw // 2 + j))
        idx_src = idx_src
        if idx_src.shape:
            if save:
                self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            for k in range(self.opt.label_nc):
                self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            if self.opt.instance_feat:
                feat = self.features_clustered[label_tgt]
                cluster_idx = self.cluster_indices[label_tgt]
                self.set_features(idx_src, feat, cluster_idx)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_objects(self, click_src, label_tgt, mask, style_id=0):
        y, x = click_src[0], click_src[1]
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
        idx_src = torch.from_numpy(mask).nonzero()
        idx_src[:, (2)] += y
        idx_src[:, (3)] += x
        self.backup_current_state()
        self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        for k in range(self.opt.label_nc):
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
        self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
        self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
        self.set_features(idx_src, self.feat, style_id)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def single_forward(self, net_input, feat_map):
        net_input = torch.cat((net_input, feat_map), dim=1)
        fake_image = self.netG.forward(net_input)
        if fake_image.size()[0] == 1:
            return fake_image.data[0]
        return fake_image.data

    def style_forward(self, click_pt, style_id=-1):
        if click_pt is None:
            self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))
            self.crop = None
            self.mask = None
        else:
            instToChange = int(self.object_map[0, 0, click_pt[0], click_pt[1]])
            self.instToChange = instToChange
            label = instToChange if instToChange < 1000 else instToChange // 1000
            self.feat = self.features_clustered[label]
            self.fake_image = []
            self.mask = self.object_map == instToChange
            idx = self.mask.nonzero()
            self.get_crop_region(idx)
            if idx.size():
                if style_id == -1:
                    min_y, min_x, max_y, max_x = self.crop
                    for cluster_idx in range(self.opt.multiple_output):
                        self.set_features(idx, self.feat, cluster_idx)
                        fake_image = self.single_forward(self.net_input, self.feat_map)
                        fake_image = util.tensor2im(fake_image[:, min_y:max_y, min_x:max_x])
                        self.fake_image.append(fake_image)
                    """### To speed up previewing different style results, either crop or downsample the label maps
                    if instToChange > 1000:
                        (min_y, min_x, max_y, max_x) = self.crop                                                
                        ### crop                                                
                        _, _, h, w = self.net_input.size()
                        offset = 512
                        y_start, x_start = max(0, min_y-offset), max(0, min_x-offset)
                        y_end, x_end = min(h, (max_y + offset)), min(w, (max_x + offset))
                        y_region = slice(y_start, y_start+(y_end-y_start)//16*16)
                        x_region = slice(x_start, x_start+(x_end-x_start)//16*16)
                        net_input = self.net_input[:,:,y_region,x_region]                    
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            fake_image = self.single_forward(net_input, self.feat_map[:,:,y_region,x_region])                            
                            fake_image = util.tensor2im(fake_image[:,min_y-y_start:max_y-y_start,min_x-x_start:max_x-x_start])
                            self.fake_image.append(fake_image)
                    else:
                        ### downsample
                        (min_y, min_x, max_y, max_x) = [crop//2 for crop in self.crop]                    
                        net_input = self.net_input[:,:,::2,::2]                    
                        size = net_input.size()
                        net_input_batch = net_input.expand(self.opt.multiple_output, size[1], size[2], size[3])             
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            feat_map = self.feat_map[:,:,::2,::2]
                            if cluster_idx == 0:
                                feat_map_batch = feat_map
                            else:
                                feat_map_batch = torch.cat((feat_map_batch, feat_map), dim=0)
                        fake_image_batch = self.single_forward(net_input_batch, feat_map_batch)
                        for i in range(self.opt.multiple_output):
                            self.fake_image.append(util.tensor2im(fake_image_batch[i,:,min_y:max_y,min_x:max_x]))"""
                else:
                    self.set_features(idx, self.feat, style_id)
                    self.cluster_indices[label] = style_id
                    self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def backup_current_state(self):
        self.net_input_prev = self.net_input.clone()
        self.label_map_prev = self.label_map.clone()
        self.inst_map_prev = self.inst_map.clone()
        self.feat_map_prev = self.feat_map.clone()

    def get_crop_region(self, idx):
        size = self.net_input.size()
        h, w = size[2], size[3]
        min_y, min_x = idx[:, (2)].min(), idx[:, (3)].min()
        max_y, max_x = idx[:, (2)].max(), idx[:, (3)].max()
        crop_min = 128
        if max_y - min_y < crop_min:
            min_y = max(0, (max_y + min_y) // 2 - crop_min // 2)
            max_y = min(h - 1, min_y + crop_min)
        if max_x - min_x < crop_min:
            min_x = max(0, (max_x + min_x) // 2 - crop_min // 2)
            max_x = min(w - 1, min_x + crop_min)
        self.crop = min_y, min_x, max_y, max_x
        self.mask = self.mask[:, :, min_y:max_y, min_x:max_x]

    def update_features(self, cluster_idx, mask=None, click_pt=None):
        self.feat_map_prev = self.feat_map.clone()
        if mask is not None:
            y, x = click_pt[0], click_pt[1]
            mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
            idx = torch.from_numpy(mask).nonzero()
            idx[:, (2)] += y
            idx[:, (3)] += x
        else:
            idx = (self.object_map == self.instToChange).nonzero()
        self.set_features(idx, self.feat, cluster_idx)

    def set_features(self, idx, feat, cluster_idx):
        for k in range(self.opt.feat_num):
            self.feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]

    def copy_features(self, idx_src, idx_tgt):
        for k in range(self.opt.feat_num):
            val = self.feat_map[idx_tgt[0], idx_tgt[1] + k, idx_tgt[2], idx_tgt[3]]
            self.feat_map[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = val

    def get_current_visuals(self, getLabel=False):
        mask = self.mask
        if self.mask is not None:
            mask = np.transpose(self.mask[0].cpu().float().numpy(), (1, 2, 0)).astype(np.uint8)
        dict_list = [('fake_image', self.fake_image), ('mask', mask)]
        if getLabel:
            label = util.tensor2label(self.net_input.data[0], self.opt.label_nc)
            dict_list += [('label', label)]
        return OrderedDict(dict_list)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (Encoder,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (LocalEnhancer,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UIModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (VGGLoss,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_Lotayou_everybody_dance_now_pytorch(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

