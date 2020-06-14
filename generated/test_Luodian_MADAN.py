import sys
_module = sys.modules[__name__]
del sys
cycada = _module
data = _module
adda_datasets = _module
bdds = _module
cityscapes = _module
cityscapes_labels = _module
cyclegan = _module
cyclegta5 = _module
cyclesynthia = _module
cyclesynthia_cyclegta5 = _module
data_loader = _module
gta5 = _module
rotater = _module
synthia = _module
util = _module
MDAN = _module
models = _module
adda_net = _module
drn = _module
fcn8s = _module
task_net = _module
util = _module
tools = _module
train_adda_net = _module
train_task_net = _module
transforms = _module
util = _module
base_data_loader = _module
base_dataset = _module
gta5_cityscapes = _module
gta_synthia_cityscapes = _module
image_folder = _module
synthia_cityscapes = _module
base_model = _module
cycle_gan_model = _module
cycle_gan_semantic_model = _module
multi_cycle_gan_semantic_model = _module
networks = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
get_data = _module
html = _module
image_pool = _module
visualizer = _module
eval_fcn = _module
train_fcn = _module
train_fcn_adda = _module
train_fcn_mdan = _module

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


import logging


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.nn import init


import math


import torch.utils.model_zoo as model_zoo


from torch import nn


from torch.autograd import Variable


from torch.utils import model_zoo


import torch.optim as optim


import logging.config


from collections import OrderedDict


from torch.nn.parameter import Parameter


import itertools


import functools


from torch.optim import lr_scheduler


from collections import deque


class GradientReversalLayer(torch.autograd.Function):
    """
	Implement the gradient reversal layer for the convenience of domain adaptation neural network.
	The forward part is the identity function while the backward part is the negative function.
	"""

    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


class MDANet(nn.Module):
    """
	Multi-layer perceptron with adversarial regularizer by domain classification.
	"""

    def __init__(self, configs):
        super(MDANet, self).__init__()
        self.pooling_layer = nn.AdaptiveAvgPool2d((2, 2))
        self.dim_reduction = nn.Conv2d(4096, 512, kernel_size=1)
        nn.init.xavier_normal_(self.dim_reduction.weight)
        nn.init.constant_(self.dim_reduction.bias, 0.1)
        self.input_dim = configs['input_dim']
        self.num_hidden_layers = len(configs['hidden_layers'])
        self.num_neurons = [] + [self.input_dim] + configs['hidden_layers']
        self.num_domains = configs['num_domains']
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.
            num_neurons[i + 1]) for i in range(self.num_hidden_layers)])
        self.softmax = nn.Linear(self.num_neurons[-1], configs['num_classes'])
        self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for
            _ in range(self.num_domains)])
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]

    def forward(self, sinputs_syn, sinputs_gta, tinputs):
        """
		:param sinputs:     A list of k inputs from k source domains.
		:param tinputs:     Input from the target domain.
		:return:
		"""
        sinputs_gta = self.pooling_layer(sinputs_gta)
        sinputs_syn = self.pooling_layer(sinputs_syn)
        tinputs = self.pooling_layer(tinputs)
        sinputs_gta = self.dim_reduction(sinputs_gta)
        sinputs_syn = self.dim_reduction(sinputs_syn)
        tinputs = self.dim_reduction(tinputs)
        b = sinputs_gta.size()[0]
        syn_relu, gta_relu, th_relu = sinputs_syn.view(b, -1
            ), sinputs_gta.view(b, -1), tinputs.view(b, -1)
        assert syn_relu[0].size()[0] == self.input_dim
        for hidden in self.hiddens:
            syn_relu = F.relu(hidden(syn_relu))
            gta_relu = F.relu(hidden(gta_relu))
        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))
        logprobs = []
        logprobs.append(F.log_softmax(self.softmax(syn_relu), dim=1))
        logprobs.append(F.log_softmax(self.softmax(gta_relu), dim=1))
        sdomains, tdomains = [], []
        sdomains.append(F.log_softmax(self.domains[0](self.grls[0](syn_relu
            )), dim=1))
        tdomains.append(F.log_softmax(self.domains[0](self.grls[0](th_relu)
            ), dim=1))
        sdomains.append(F.log_softmax(self.domains[1](self.grls[1](gta_relu
            )), dim=1))
        tdomains.append(F.log_softmax(self.domains[1](self.grls[1](th_relu)
            ), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs


models = {}


def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def register_model(name):

    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


@register_model('AddaNet')
class AddaNet(nn.Module):
    """Defines and Adda Network."""

    def __init__(self, num_cls=10, model='LeNet', src_weights_init=None,
        weights_init=None):
        super(AddaNet, self).__init__()
        self.name = 'AddaNet'
        self.base_model = model
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init)
        elif src_weights_init is not None:
            self.load_src_net(src_weights_init)
        else:
            raise Exception('AddaNet must be initialized with weights.')

    def forward(self, x_s, x_t):
        """Pass source and target images through their
        respective networks."""
        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)
        if self.discrim_feat:
            d_s = self.discriminator(x_s)
            d_t = self.discriminator(x_t)
        else:
            d_s = self.discriminator(score_s)
            d_t = self.discriminator(score_t)
        return score_s, score_t, d_s, d_t

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.src_net = get_model(self.base_model, num_cls=self.num_cls)
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls)
        input_dim = self.num_cls
        self.discriminator = nn.Sequential(nn.Linear(input_dim, 500), nn.
            ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2))
        self.image_size = self.src_net.image_size
        self.num_channels = self.src_net.num_channels

    def load(self, init_path):
        """Loads full src and tgt models."""
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def load_src_net(self, init_path):
        """Initialize source and target with source
        weights."""
        self.src_net.load(init_path)
        self.tgt_net.load(init_path)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


model_urls = {'drn26':
    'https://tigress-web.princeton.edu/~fy/drn/models/drn26-ddedf421.pth',
    'drn42':
    'https://tigress-web.princeton.edu/~fy/drn/models/drn42-9d336e8c.pth',
    'drn58':
    'https://tigress-web.princeton.edu/~fy/drn/models/drn58-0a53a92c.pth'}


def safe_load_state_dict(net, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. Any params in :attr:`state_dict`
    that do not match the keys returned by :attr:`net`'s :func:`state_dict()`
    method or have differing sizes are skipped.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = net.state_dict()
    skipped = []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue
        if isinstance(param, Parameter):
            param = param.data
        if own_state[name].size() != param.size():
            skipped.append(name)
            continue
        own_state[name].copy_(param)
    if skipped:
        logging.info('Skipped loading some parameters: {}'.format(skipped))


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    return torch.from_numpy(filter).float()


class Bilinear(nn.Module):

    def __init__(self, factor, num_channels):
        super().__init__()
        self.factor = factor
        filter = get_upsample_filter(factor * 2)
        w = torch.zeros(num_channels, num_channels, factor * 2, factor * 2)
        for i in range(num_channels):
            w[i, i] = filter
        self.register_buffer('w', w)

    def forward(self, x):
        return F.conv_transpose2d(x, Variable(self.w), stride=self.factor)


def _crop(input, shape, offset=0):
    _, _, h, w = shape.size()
    return input[:, :, offset:offset + h, offset:offset + w].contiguous()


def make_layers(cfg, batch_norm=False):
    """This is almost verbatim from torchvision.models.vgg, except that the
	MaxPool2d modules are configured with ceil_mode=True.
	"""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                )
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            modules = [conv2d, nn.ReLU(inplace=True)]
            if batch_norm:
                modules.insert(1, nn.BatchNorm2d(v))
            layers.extend(modules)
            in_channels = v
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, input_dim=4096, output_dim=2, pretrained=False,
        weights_init=''):
        super().__init__()
        dim1 = 1024 if input_dim == 4096 else 512
        dim2 = int(dim1 / 2)
        self.D = nn.Sequential(nn.Conv2d(input_dim, dim1, 1), nn.Dropout2d(
            p=0.5), nn.ReLU(inplace=True), nn.Conv2d(dim1, dim2, 1), nn.
            Dropout2d(p=0.5), nn.ReLU(inplace=True), nn.Conv2d(dim2,
            output_dim, 1))
        if pretrained and weights_init is not None:
            self.load_weights(weights_init)

    def forward(self, x):
        d_score = self.D(x)
        return d_score

    def load_weights(self, weights):
        None
        self.load_state_dict(torch.load(weights))


def init_eye(tensor):
    if isinstance(tensor, Variable):
        init_eye(tensor.data)
        return tensor
    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


class Transform_Module(nn.Module):

    def __init__(self, input_dim=4096):
        super().__init__()
        self.transform = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_eye(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        t_x = self.transform(x)
        return t_x


def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or 
            classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type
                    )
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class TaskNet(nn.Module):
    num_channels = 3
    image_size = 32
    name = 'TaskNet'
    """Basic class which does classification."""

    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        score = self.classifier(x)
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
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
        self.model = nn.Sequential(*model)

    def forward(self, input):
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


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=
        nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
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
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1,
            padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2,
            kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(
            ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1,
            kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Classifier(nn.Module):

    def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()
        kw = 3
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2), nn.
            LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2), norm_layer(ndf * nf_mult, affine
                =True), nn.LeakyReLU(0.2, True)]
        self.before_linear = nn.Sequential(*sequence)
        sequence = [nn.Linear(ndf * nf_mult, 1024), nn.Linear(1024, 10)]
        self.after_linear = nn.Sequential(*sequence)

    def forward(self, x):
        bs = x.size(0)
        out = self.after_linear(self.before_linear(x).view(bs, -1))
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Luodian_MADAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Bilinear(*[], **{'factor': 4, 'num_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 4096, 64, 64])], {})

    def test_003(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(PixelDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(UnetGenerator(*[], **{'input_nc': 4, 'output_nc': 4, 'num_downs': 4}), [torch.rand([4, 4, 64, 64])], {})

