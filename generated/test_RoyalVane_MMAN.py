import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
single_dataset = _module
unaligned_dataset = _module
models = _module
base_model = _module
cycle_gan_model = _module
deeplab = _module
focal_loss = _module
networks = _module
pix2pix_model = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
get_data = _module
html = _module
image_pool = _module
png = _module
visualizer = _module

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


import random


import torch


import torch.nn as nn


import math


import numpy as np


from collections import OrderedDict


from torch.autograd import Variable


import itertools


import torch.nn.functional as F


from torch.nn import init


import functools


from torch.optim import lr_scheduler


class D_Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(D_Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv1_1 = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        self.conv1_1.weight.data.normal_(0, 0.01)
        conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.conv2d_list.append(conv1)
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes,
                kernel_size=3, stride=1, padding=padding, dilation=dilation,
                bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat([out, self.conv2d_list[i + 1](x)], 1)
        out = self.conv1_1(out)
        return out


affine_par = True


class D_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(D_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class D_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, input_size):
        self.inplanes = 64
        super(D_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
            dilation=[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilation=[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilation=[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=[1, 1, 1], rate=2)
        self.layer5 = self._make_pred_layer(D_Classifier_Module, [2, 4, 6],
            [2, 4, 6], num_classes)
        self.upsample = nn.Upsample(input_size, mode='bilinear')

    def _make_layer(self, block, planes, blocks, stride=1, dilation=[1], rate=1
        ):
        downsample = None
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.
            expansion, kernel_size=1, stride=stride, bias=False), nn.
            BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=rate *
            dilation[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=rate *
                dilation[i] if len(dilation) > 1 else dilation[0]))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        sm = nn.Softmax2d()
        lsm = nn.LogSoftmax()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.upsample(x)
        return {'GAN': x, 'L1': lsm(x)}


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \\alpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        H = inputs.size(2)
        W = inputs.size(3)
        sm = nn.Softmax2d()
        P = sm(inputs)
        class_mask = inputs.data.new(N, C, H, W).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(N, 1, H, W)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(N, H, W)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0, tensor=torch.FloatTensor):
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
            create_label = (self.real_label_var is None or self.
                real_label_var.numel() != input.numel())
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False
                    )
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None or self.
                fake_label_var.numel() != input.numel())
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False
                    )
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
            kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.
            ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim), nn.ReLU(True)]
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
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Bottleneck(nn.Module):

    def __init__(self, model_cx, model_x):
        super(Bottleneck, self).__init__()
        self.ReLU = nn.ReLU()
        self.model_cx = self.build_block(model_cx)
        self.model_x = self.build_block(model_x)

    def build_block(self, model):
        return nn.Sequential(*model)

    def forward(self, x):
        x = self.ReLU(x)
        if len(self.model_x) == 0:
            return self.model_cx(x) + x
        else:
            return self.model_cx(x) + self.model_x(x)


class ASPP_Module(nn.Module):

    def __init__(self, input_nc, conv2d_list):
        super(ASPP_Module, self).__init__()
        self.conv2d_list = conv2d_list
        self.conv1_1 = nn.Conv2d(input_nc * 4, input_nc, kernel_size=1)
        self.conv1_1.weight.data.normal_(0, 0.01)
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat([out, self.conv2d_list[i + 1](x)], 1)
        out = self.conv1_1(out)
        return out


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, hook, ngf=64,
        norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        model_res101 = models.resnet101(pretrained=True)
        model_res101 = model_res101
        T_block = UnetSkipConnectionBlock(output_nc, ngf * 32, input_nc=ngf *
            32, submodule=None, depth=-2, norm_layer=norm_layer, model_ft=
            model_res101)
        handle = T_block.register_forward_hook(hook.hook_out)
        U_block = UnetSkipConnectionBlock(output_nc, ngf * 32, input_nc=
            None, submodule=T_block, depth=-1, norm_layer=norm_layer,
            model_ft=model_res101)
        U_block = UnetSkipConnectionBlock(ngf * 16, ngf * 32, input_nc=None,
            submodule=U_block, depth=0, norm_layer=norm_layer, model_ft=
            model_res101)
        U_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None,
            submodule=U_block, depth=1, norm_layer=norm_layer, model_ft=
            model_res101)
        U_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
            submodule=U_block, depth=2, norm_layer=norm_layer, model_ft=
            model_res101)
        U_block = UnetSkipConnectionBlock(ngf, ngf * 4, input_nc=None,
            submodule=U_block, depth=3, norm_layer=norm_layer, model_ft=
            model_res101)
        U_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
            submodule=U_block, depth=4, norm_layer=norm_layer, model_ft=
            model_res101)
        self.model = U_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, depth, input_nc=None, submodule=
        None, norm_layer=nn.BatchNorm2d, use_dropout=False, model_ft=None):
        super(UnetSkipConnectionBlock, self).__init__()
        assert depth <= 4
        self.depth = depth
        ResBlock0 = [model_ft.conv1, model_ft.bn1]
        ResBlock1 = [model_ft.maxpool]
        for i in range(3):
            model_x = []
            model_cx = []
            if i == 0:
                model_x = [model_ft.layer1[i].downsample[0], model_ft.
                    layer1[i].downsample[1]]
            model_cx = [model_ft.layer1[i].conv1, model_ft.layer1[i].bn1,
                model_ft.layer1[i].conv2, model_ft.layer1[i].bn2, model_ft.
                layer1[i].conv3, model_ft.layer1[i].bn3]
            ResBlock1 += [Bottleneck(model_cx, model_x)]
        ResBlock2 = []
        for j in range(4):
            model_x = []
            model_cx = []
            if j == 0:
                model_x = [model_ft.layer2[j].downsample[0], model_ft.
                    layer2[j].downsample[1]]
            model_cx = [model_ft.layer2[j].conv1, model_ft.layer2[j].bn1,
                model_ft.layer2[j].conv2, model_ft.layer2[j].bn2, model_ft.
                layer2[j].conv3, model_ft.layer2[j].bn3]
            ResBlock2 += [Bottleneck(model_cx, model_x)]
        ResBlock3 = []
        for k in range(23):
            model_x = []
            model_cx = []
            if k == 0:
                model_x = [model_ft.layer3[k].downsample[0], model_ft.
                    layer3[k].downsample[1]]
            model_cx = [model_ft.layer3[k].conv1, model_ft.layer3[k].bn1,
                model_ft.layer3[k].conv2, model_ft.layer3[k].bn2, model_ft.
                layer3[k].conv3, model_ft.layer3[k].bn3]
            ResBlock3 += [Bottleneck(model_cx, model_x)]
        ResBlock4 = []
        for m in range(3):
            model_x = []
            model_cx = []
            if m == 0:
                model_x = [model_ft.layer4[m].downsample[0], model_ft.
                    layer4[m].downsample[1]]
                model_x[0].stride = 1, 1
            model_cx = [model_ft.layer4[m].conv1, model_ft.layer4[m].bn1,
                model_ft.layer4[m].conv2, model_ft.layer4[m].bn2, model_ft.
                layer4[m].conv3, model_ft.layer4[m].bn3]
            model_cx[2].stride = 1, 1
            model_cx[2].dilation = 2, 2
            model_cx[2].padding = 2, 2
            ResBlock4 += [Bottleneck(model_cx, model_x)]
        ResBlock5 = []
        conv_list = nn.ModuleList()
        conv1 = nn.Conv2d(inner_nc, outer_nc, kernel_size=1)
        conv_list.append(conv1)
        for n in range(1, 4):
            conv3 = nn.Conv2d(inner_nc, outer_nc, kernel_size=3)
            conv3.stride = 1, 1
            conv3.dilation = 2 * n, 2 * n
            conv3.padding = 2 * n, 2 * n
            conv_list.append(conv3)
        ResBlock5 += [ASPP_Module(outer_nc, conv_list)]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)
        if depth == 4:
            down = ResBlock0
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            up = [uprelu, upconv]
            model = down + [submodule] + up
            self.U4 = nn.Sequential(*model)
        if depth == 3:
            down = ResBlock1
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U3 = nn.Sequential(*model)
            self.con3 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con3.weight.data.normal_(0, 0.01)
        if depth == 2:
            down = ResBlock2
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U2 = nn.Sequential(*model)
            self.con2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con2.weight.data.normal_(0, 0.01)
        if depth == 1:
            down = ResBlock3
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U1 = nn.Sequential(*model)
            self.con1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con1.weight.data.normal_(0, 0.01)
        if depth == 0:
            down = ResBlock4
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=3,
                stride=1, padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U0 = nn.Sequential(*model)
            self.con0 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con0.weight.data.normal_(0, 0.01)
        if depth == -1:
            model = [submodule]
            self.U_1 = nn.Sequential(*model)
        if depth == -2:
            down = ResBlock5
            lsm = [nn.LogSoftmax()]
            model = down + lsm
            self.U_2 = nn.Sequential(*model)

    def forward(self, x):
        if self.depth == 4:
            sm = nn.Softmax2d()
            lsm = nn.LogSoftmax()
            t = self.U4(x)
            return {'GAN': sm(t * 5.0), 'L1': lsm(t)}
        elif self.depth == 3:
            return torch.cat([self.con3(x), self.U3(x)], 1)
        elif self.depth == 2:
            return torch.cat([self.con2(x), self.U2(x)], 1)
        elif self.depth == 1:
            return torch.cat([self.con1(x), self.U1(x)], 1)
        elif self.depth == 0:
            return torch.cat([self.con0(x), self.U0(x)], 1)
        elif self.depth == -1:
            _ = self.U_1(x)
            return x
        elif self.depth == -2:
            sm = nn.Softmax2d()
            lsm = nn.LogSoftmax()
            t = self.U_2(x)
            return {'GAN': sm(t * 5.0), 'L1': lsm(t)}


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = False
        else:
            use_bias = False
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_RoyalVane_MMAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(GANLoss(*[], **{}), [], {'input': torch.rand([4, 4]), 'target_is_real': 4})

    @_fails_compile()
    def test_002(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

