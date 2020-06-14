import sys
_module = sys.modules[__name__]
del sys
lib = _module
Dataset_Base = _module
datasets = _module
dataset_naiReg = _module
eval = _module
eval_aet_multilevel = _module
regEulerNet = _module
trainval_workdir = _module
dataengine_v2 = _module
eval_official = _module
eval_sn = _module
helper = _module
model = _module
modelSE = _module
modelSEv2 = _module
regNormalNet = _module
trainval_workdir = _module
test = _module
unet_model = _module
unet_parts = _module
dataset_regQuatNet = _module
db_type = _module
design = _module
eval_quat_multilevel = _module
helper = _module
regQuatNet = _module
trainval_workdir = _module
Pascal3D = _module
anno_db_v2 = _module
util_v2 = _module
data_info = _module
build_imdb = _module
fix_bad_bbox_anno = _module
gen_gt_box = _module
mat2py = _module
prepare_anno_db = _module
prepare_r4cnn_syn_data = _module
basic = _module
common = _module
util = _module
yaml_netconf = _module
lmdb_util = _module
imagedata_lmdb = _module
load_nested_mat_struct = _module
numpy_db = _module
formater = _module
numpy_db_py3 = _module
ordered_easydict = _module
pytorch_util = _module
libtrain = _module
init_torch = _module
reducer = _module
reducer_v2 = _module
tools = _module
netutil = _module
common_v2 = _module
trunk_alexnet_bvlc = _module
trunk_alexnet_pytorch = _module
trunk_inception = _module
trunk_resnet = _module
trunk_vgg = _module
torch_3rd_funcs = _module
torch_3rd_layers = _module
torch_v4_feature = _module
txt_table_v1 = _module

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


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


import torch


import numpy as np


from collections import OrderedDict as odict


from itertools import product


import torch.optim


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import BatchSampler


from math import pi


import math


import re


from string import Template


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn.modules.module import Module


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.parameter import Parameter


from torch.nn.functional import *


class down(nn.Module):

    def __init__(self, block_cfg=['M', 64, 64], in_channels=3, batch_norm=True
        ):
        super(down, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2,
                    return_indices=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.conv = nn.Sequential(*layers)
        if block_cfg[0] == 'M':
            self.pool = layers[0]
            self.conv = nn.Sequential(*layers[1:])
        else:
            self.pool = None
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.pool is not None:
            x, poolInd = self.pool(x)
            x = self.conv(x)
            return x, poolInd
        else:
            x = self.conv(x)
            return x


class up(nn.Module):

    def __init__(self, block_cfg, in_channels, batch_norm=True):
        super(up, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3,
                    padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if block_cfg[-1] == 'M':
            self.deconv = nn.Sequential(*layers[:-1])
            self.uppool = layers[-1]
        else:
            self.deconv = nn.Sequential(*layers)

    def forward(self, x1, poolInd=None, x2=None):
        """First deconv and then do uppool,cat if needed."""
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = self.uppool(x1, poolInd, output_size=x2.size())
            x1 = torch.cat([x2, x1], dim=1)
        return x1


def init_weights_by_filling(nn_module_or_seq, gaussian_std=0.01,
    kaiming_normal=True, silent=False):
    """ Note: gaussian_std is fully connected layer (nn.Linear) only.
        For nn.Conv2d:
           If kaiming_normal is enable, nn.Conv2d is initialized by kaiming_normal.
           Otherwise, initialized based on kernel size.
    """
    if not silent:
        print(
            '[init_weights_by_filling]  gaussian_std=%s   kaiming_normal=%s \n %s'
             % (gaussian_std, kaiming_normal, nn_module_or_seq))
    for name, m in nn_module_or_seq.named_modules():
        if isinstance(m, nn.Conv2d):
            if kaiming_normal:
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0, std=gaussian_std)
            m.bias.data.zero_()
    return nn_module_or_seq


_global_config['torchmodel'] = 4


def _copy_weights_from_torchmodel(own_state, pretrained_type='alexnet',
    strict=True, src2dsts=None):
    from torch.nn.parameter import Parameter
    print('-----------------------')
    print('[Info] Copy from  %s ' % pretrained_type)
    print('-----------------------')
    src_state = model_zoo.load_url(cfg.torchmodel[pretrained_type].model_url)
    not_copied = list(own_state.keys())
    if src2dsts is not None:
        for src, dsts in src2dsts.items():
            if not isinstance(dsts, list):
                dsts = [dsts]
            for dst in dsts:
                if dst in own_state.keys():
                    own_state[dst].copy_(src_state[src])
                    not_copied.remove(dst)
                    print('%-20s  -->  %-20s' % (src, dst))
                else:
                    dst_w_name, src_w_name = ('%s.weight' % dst, 
                        '%s.weight' % src)
                    dst_b_name, src_b_name = '%s.bias' % dst, '%s.bias' % src
                    if (dst_w_name not in own_state.keys() or dst_b_name not in
                        own_state.keys()) and not strict:
                        print('%-20s  -->  %-20s   [ignored] Missing dst.' %
                            (src, dst))
                        continue
                    print('%-20s  -->  %-20s' % (src, dst))
                    assert own_state[dst_w_name].shape == src_state[src_w_name
                        ].shape, '[%s] w: dest. %s != src. %s' % (dst_w_name,
                        own_state[dst_w_name].shape, src_state[src_w_name].
                        shape)
                    own_state[dst_w_name].copy_(src_state[src_w_name])
                    not_copied.remove(dst_w_name)
                    assert own_state[dst_b_name].shape == src_state[src_b_name
                        ].shape, '[%s] w: dest. %s != src. %s' % (dst_b_name,
                        own_state[dst_b_name].shape, src_state[src_b_name].
                        shape)
                    own_state[dst_b_name].copy_(src_state[src_b_name])
                    not_copied.remove(dst_b_name)
    else:
        for name, param in src_state.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    print('%-30s  -->  %-30s' % (name, name))
                    own_state[name].copy_(param)
                    not_copied.remove(name)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                        .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name)
                    )
            else:
                print('%-30s  -->  %-30s   [ignored] Missing dst.' % (name,
                    name))
    for name in not_copied:
        if name.endswith('.weight'):
            own_state[name].normal_(mean=0.0, std=0.005)
            print('%-20s  -->  %-20s' % ('[filler] gaussian005', name))
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            print('%-30s  -->  %-30s' % ('[filler] 0', name))
        elif name.endswith('.running_mean') or name.endswith('.running_var'
            ) or name.endswith('num_batches_tracked'):
            print('*************************** pass', name)
        else:
            print('Unknow parameter type: ', name)
            raise NotImplementedError
    if strict:
        missing = set(own_state.keys()) - set(src_state.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    print('-----------------------')


def get_weights_from_caffesnapeshot(proto_file, model_file):
    import caffe
    from collections import OrderedDict
    import cPickle as pickle
    caffe.set_mode_cpu()
    net = caffe.Net(proto_file, model_file, caffe.TEST)
    model_dict = OrderedDict()
    for layer_name, param in net.params.iteritems():
        learnable_weight = []
        if len(param) == 2:
            learnable_weight.append(param[0].data.copy())
            learnable_weight.append(param[1].data.copy())
        elif len(param) == 1:
            learnable_weight.append(param[0].data.copy())
        else:
            raise NotImplementedError
        model_dict[layer_name] = learnable_weight
    return model_dict


_global_config['caffemodel'] = 4


def _copy_weights_from_caffemodel(own_state, pretrained_type='alexnet',
    ignore_missing_dst=False, src2dsts=dict(conv1='conv1', conv2='conv2',
    conv3='conv3', conv4='conv4', conv5='conv5', fc6='fc6', fc7='fc7')):
    """ src2dsts = dict(conv1='conv1', conv2='conv2', conv3='conv3', conv4='conv4', conv5='conv5')
    Or in list:
        src2dsts = dict(conv1=['conv1'], conv2=['conv2'], conv3=['conv3'], conv4=['conv4'], conv5=['conv5'])
    """
    print('-----------------------')
    print('[Info] Copy from  %s ' % pretrained_type)
    print('-----------------------')
    if isinstance(pretrained_type, tuple):
        proto_file, model_file = pretrained_type
        pretrained_weights = get_weights_from_caffesnapeshot(proto_file,
            model_file)
    else:
        import pickle
        print('Loading: %s' % cfg.caffemodel[pretrained_type].pkl)
        pretrained_weights = pickle.load(open(cfg.caffemodel[
            pretrained_type].pkl, 'rb'), encoding='bytes')
        print(pretrained_weights.keys())
    not_copied = list(own_state.keys())
    src_list = sorted(src2dsts.keys())
    for src in src_list:
        dsts = src2dsts[src]
        if not isinstance(dsts, list):
            dsts = [dsts]
        w, b = pretrained_weights[src.encode('utf-8')]
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)
        for dst in dsts:
            if ignore_missing_dst and dst not in own_state.keys():
                print('%-20s  -->  %-20s   [ignored] Missing dst.' % (src, dst)
                    )
                continue
            print('%-20s  -->  %-20s' % (src, dst))
            dst_w_name = '%s.weight' % dst
            dst_b_name = '%s.bias' % dst
            assert dst_w_name in own_state.keys(), '[Error] %s not in %s' % (
                dst_w_name, own_state.keys())
            assert dst_b_name in own_state.keys(), '[Error] %s not in %s' % (
                dst_b_name, own_state.keys())
            assert own_state[dst_w_name
                ].shape == w.shape, '[%s] w: dest. %s != src. %s' % (dst_w_name
                , own_state[dst_w_name].shape, w.shape)
            own_state[dst_w_name].copy_(w)
            not_copied.remove(dst_w_name)
            assert own_state[dst_b_name
                ].shape == b.shape, '[%s] w: dest. %s != src. %s' % (dst_b_name
                , own_state[dst_b_name].shape, b.shape)
            own_state[dst_b_name].copy_(b)
            not_copied.remove(dst_b_name)
    for name in not_copied:
        if name.endswith('.weight'):
            own_state[name].normal_(mean=0.0, std=0.005)
            print('%-20s  -->  %-20s' % ('[filler] gaussian005', name))
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            print('%-20s  -->  %-20s' % ('[filler] 0', name))
        else:
            print('Unknow parameter type: ', name)
            raise NotImplementedError
    print('-----------------------')


def copy_weights(own_state, pretrained_type, **kwargs):
    """ Usage example:
            copy_weights(own_state, 'torchmodel.alexnet',  strict=False)
            copy_weights(own_state, 'caffemodel.alexnet',  ignore_missing_dst=True, src2dsts={})
            copy_weights(own_state, '')
    """
    if isinstance(pretrained_type, str):
        if pretrained_type.startswith('torchmodel.'):
            pretrained_type = pretrained_type[11:]
            copy_func = _copy_weights_from_torchmodel
        elif pretrained_type.startswith('caffemodel.'):
            pretrained_type = pretrained_type[11:]
            copy_func = _copy_weights_from_caffemodel
        else:
            print('Unkonw pretrained_type: ', pretrained_type)
            raise NotImplementedError
    elif isinstance(pretrained_type, tuple):
        copy_func = _copy_weights_from_caffemodel
    else:
        print('Unkonw type(pretrained_type): ', type(pretrained_type))
        raise NotImplementedError
    copy_func(own_state, pretrained_type, **kwargs)


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
        dtype=np.float64)
    weight[(range(in_channels)), (range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class down(nn.Module):

    def __init__(self, block_cfg=['M', 64, 64], in_channels=3, batch_norm=True
        ):
        super(down, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2,
                    return_indices=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.conv = nn.Sequential(*layers)
        if block_cfg[0] == 'M':
            self.pool = layers[0]
            self.conv = nn.Sequential(*layers[1:])
        else:
            self.pool = None
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.pool is not None:
            x, poolInd = self.pool(x)
            x = self.conv(x)
            return x, poolInd
        else:
            x = self.conv(x)
            return x


class up(nn.Module):

    def __init__(self, block_cfg, in_channels, batch_norm=True):
        super(up, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3,
                    padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if block_cfg[-1] == 'M':
            self.deconv = nn.Sequential(*layers[:-1])
            self.uppool = layers[-1]
        else:
            self.deconv = nn.Sequential(*layers)

    def forward(self, x1, poolInd=None, x2=None):
        """First deconv and then do uppool,cat if needed."""
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = self.uppool(x1, poolInd, output_size=x2.size())
            x1 = torch.cat([x2, x1], dim=1)
        return x1


class VGG16_Trunk(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG16_Trunk, self).__init__()
        self.net_arch = 'vgg16_bn'
        self.conv1 = down([64, 64], in_channels=3)
        self.conv2 = down(['M', 128, 128], in_channels=64)
        self.conv3 = down(['M', 256, 256, 256], in_channels=128)
        self.conv4 = down(['M', 512, 512, 512], in_channels=256)
        self.conv5 = down(['M', 512, 512, 512], in_channels=512)
        self.deconv5 = up([512, 512, 512, 'M'], in_channels=512)
        self.deconv4 = up([512, 512, 256, 'M'], in_channels=512 + 512)
        self.deconv3 = up([256, 256, 128, 'M'], in_channels=256 + 256)
        self.deconv2 = up([128, 64, 'M'], in_channels=128 + 128)
        self.deconv1_abs = up([64, 3], in_channels=64 + 64)
        self.deconv1_sgc = up([64, 4], in_channels=64 + 64)
        self.deconv1_abs.deconv = self.deconv1_abs.deconv[:-1]
        self.deconv1_sgc.deconv = self.deconv1_sgc.deconv[:-1]
        if init_weights:
            self.init_weights('torchmodel')

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        en1 = self.conv1(x)
        en2, poolInd1 = self.conv2(en1)
        en3, poolInd2 = self.conv3(en2)
        en4, poolInd3 = self.conv4(en3)
        en5, poolInd4 = self.conv5(en4)
        de4 = self.deconv5(en5, poolInd4, en4)
        de3 = self.deconv4(de4, poolInd3, en3)
        de2 = self.deconv3(de3, poolInd2, en2)
        de1 = self.deconv2(de2, poolInd1, en1)
        x_abs = self.deconv1_abs(de1)
        x_sgc = self.deconv1_sgc(de1)
        return x_abs, x_sgc

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self)
        elif pretrained == 'caffemodel':
            None
            src2dsts = odict(conv1_1='conv1.conv.0', conv1_2='conv1.conv.2',
                conv2_1='conv2.conv.0', conv2_2='conv2.conv.2', conv3_1=
                'conv3.conv.0', conv3_2='conv3.conv.2', conv3_3=
                'conv3.conv.4', conv4_1='conv4.conv.0', conv4_2=
                'conv4.conv.2', conv4_3='conv4.conv.4', conv5_1=
                'conv5.conv.0', conv5_2='conv5.conv.2', conv5_3='conv5.conv.4')
            copy_weights(self.state_dict(), 'caffemodel.%s' % self.net_arch,
                src2dsts=src2dsts)
        elif pretrained == 'torchmodel':
            src2dsts = odict([map(str.strip, x.split('=')) for x in map(str
                .strip, open(this_dir + '/src2dsts.vgg16_bn.txt').readlines
                ()) if not x.startswith('=')])
            copy_weights(self.state_dict(), 'torchmodel.%s' % self.net_arch,
                strict=False, src2dsts=src2dsts)
        else:
            None
            raise NotImplementedError
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                None
                assert m.kernel_size[0] == m.kernel_size[1]
                try:
                    initial_weight = get_upsampling_weight(m.in_channels, m
                        .out_channels, m.kernel_size[0])
                    None
                    m.weight.data.copy_(initial_weight)
                except:
                    None
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                None
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        return self

    def fix_conv1_conv2(self):
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


class down(nn.Module):

    def __init__(self, block_cfg=['M', 64, 64], in_channels=3, batch_norm=True
        ):
        super(down, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2,
                    return_indices=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.conv = nn.Sequential(*layers)
        if block_cfg[0] == 'M':
            self.pool = layers[0]
            self.conv = nn.Sequential(*layers[1:])
        else:
            self.pool = None
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.pool is not None:
            x, poolInd = self.pool(x)
            x = self.conv(x)
            return x, poolInd
        else:
            x = self.conv(x)
            return x


class up(nn.Module):

    def __init__(self, block_cfg, in_channels, batch_norm=True):
        super(up, self).__init__()
        layers = []
        for v in block_cfg:
            if v == 'M':
                layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3,
                    padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if block_cfg[-1] == 'M':
            self.deconv = nn.Sequential(*layers[:-1])
            self.uppool = layers[-1]
        else:
            self.deconv = nn.Sequential(*layers)

    def forward(self, x1, poolInd=None, x2=None):
        """First deconv and then do uppool,cat if needed."""
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = self.uppool(x1, poolInd, output_size=x2.size())
            x1 = torch.cat([x2, x1], dim=1)
        return x1


class VGG16_Trunk(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG16_Trunk, self).__init__()
        self.net_arch = 'vgg16'
        self.conv1 = down([64, 64], in_channels=3)
        self.conv2 = down(['M', 128, 128], in_channels=64)
        self.conv3 = down(['M', 256, 256, 256], in_channels=128)
        self.conv4 = down(['M', 512, 512, 512], in_channels=256)
        self.conv5 = down(['M', 512, 512, 512], in_channels=512)
        self.deconv5_abs = up([512, 512, 512, 'M'], in_channels=512)
        self.deconv4_abs = up([512, 512, 256, 'M'], in_channels=512 + 512)
        self.deconv3_abs = up([256, 256, 128, 'M'], in_channels=256 + 256)
        self.deconv2_abs = up([128, 64, 'M'], in_channels=128 + 128)
        self.deconv1_abs = up([64, 3], in_channels=64 + 64)
        self.deconv5_sgc = up([512, 512, 512, 'M'], in_channels=512)
        self.deconv4_sgc = up([512, 512, 256, 'M'], in_channels=512 + 512)
        self.deconv3_sgc = up([256, 256, 128, 'M'], in_channels=256 + 256)
        self.deconv2_sgc = up([128, 64, 'M'], in_channels=128 + 128)
        self.deconv1_sgc = up([64, 4], in_channels=64 + 64)
        self.deconv1_abs.deconv = self.deconv1_abs.deconv[:-1]
        self.deconv1_sgc.deconv = self.deconv1_sgc.deconv[:-1]
        if init_weights:
            self.init_weights(None)

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        en1 = self.conv1(x)
        en2, poolInd1 = self.conv2(en1)
        en3, poolInd2 = self.conv3(en2)
        en4, poolInd3 = self.conv4(en3)
        en5, poolInd4 = self.conv5(en4)
        de4_abs = self.deconv5_abs(en5, poolInd4, en4)
        de3_abs = self.deconv4_abs(de4_abs, poolInd3, en3)
        de2_abs = self.deconv3_abs(de3_abs, poolInd2, en2)
        de1_abs = self.deconv2_abs(de2_abs, poolInd1, en1)
        x_abs = self.deconv1_abs(de1_abs)
        de4_sgc = self.deconv5_sgc(en5, poolInd4, en4)
        de3_sgc = self.deconv4_sgc(de4_sgc, poolInd3, en3)
        de2_sgc = self.deconv3_sgc(de3_sgc, poolInd2, en2)
        de1_sgc = self.deconv2_sgc(de2_sgc, poolInd1, en1)
        x_sgc = self.deconv1_sgc(de1_sgc)
        return x_abs, x_sgc

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self)
        elif pretrained == 'caffemodel':
            None
            src2dsts = odict(conv1_1='conv1.conv.0', conv1_2='conv1.conv.2',
                conv2_1='conv2.conv.0', conv2_2='conv2.conv.2', conv3_1=
                'conv3.conv.0', conv3_2='conv3.conv.2', conv3_3=
                'conv3.conv.4', conv4_1='conv4.conv.0', conv4_2=
                'conv4.conv.2', conv4_3='conv4.conv.4', conv5_1=
                'conv5.conv.0', conv5_2='conv5.conv.2', conv5_3='conv5.conv.4')
            copy_weights(self.state_dict(), 'caffemodel.%s' % self.net_arch,
                src2dsts=src2dsts)
        elif pretrained == 'torchmodel':
            src2dsts = odict([('features.0', 'conv1.conv.0'), ('features.2',
                'conv1.conv.2'), ('features.5', 'conv2.conv.0'), (
                'features.7', 'conv2.conv.2'), ('features.10',
                'conv3.conv.0'), ('features.12', 'conv3.conv.2'), (
                'features.14', 'conv3.conv.4'), ('features.17',
                'conv4.conv.0'), ('features.19', 'conv4.conv.2'), (
                'features.21', 'conv4.conv.4'), ('features.24',
                'conv5.conv.0'), ('features.26', 'conv5.conv.2'), (
                'features.28', 'conv5.conv.4')])
            copy_weights(self.state_dict(), 'torchmodel.%s' % self.net_arch,
                strict=False, src2dsts=src2dsts)
        else:
            raise NotImplementedError
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                None
                assert m.kernel_size[0] == m.kernel_size[1]
                try:
                    initial_weight = get_upsampling_weight(m.in_channels, m
                        .out_channels, m.kernel_size[0])
                    None
                    m.weight.data.copy_(initial_weight)
                except:
                    None
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                None
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        return self

    def fix_conv1_conv2(self):
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


class _regNormalNet(nn.Module):

    def __init__(self, method, net_arch='vgg16', init_weights=True):
        super(_regNormalNet, self).__init__()
        _Trunk = net_arch2Trunk[net_arch][method]
        self.trunk = _Trunk(init_weights=init_weights)

    def forword(self, x, label):
        raise NotImplementedError


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x_ = self.outc(x1_)
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        return x


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch,
            out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=
            True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2))
            )
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class _BaseReg_Net(nn.Module):

    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334, init_weights=True):
        seq = nn.Sequential(nn.Linear(in_size, nr_fc8), nn.ReLU(inplace=
            True), nn.Linear(nr_fc8, nr_cate * reg_n_D))
        if init_weights:
            init_weights_by_filling(seq, gaussian_std=0.005, kaiming_normal
                =True)
        return seq
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        super(_BaseReg_Net, self).__init__()
        _Trunk = net_arch2Trunk[net_arch]
        self.trunk = _Trunk(init_weights=init_weights)
        self.nr_cate = nr_cate
        self.top_size = 4096 if not self.trunk.net_arch.startswith('resnet'
            ) else 2048

    def forword(self, x, label):
        raise NotImplementedError


class AlexNet_Trunk(nn.Module):

    def __init__(self, init_weights=True):
        super(AlexNet_Trunk, self).__init__()
        self.net_arch = 'alexnet'
        self.Convs = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=
            4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=
            3, stride=2), LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1
            ), nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), nn.
            ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
            LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1), nn.Conv2d(
            256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(384, 384, kernel_size=3, padding=1, groups=2), nn.ReLU(
            inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1,
            groups=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.Fcs = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(
            inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(
            inplace=True), nn.Dropout())
        if init_weights == True:
            self.init_weights(pretrained='caffemodel')
        elif isinstance(init_weights, str) or init_weights is None:
            self.init_weights(pretrained=init_weights)
        else:
            raise NotImplementedError

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 227, 227)
        """
        x = self.Convs(x)
        batchsize = x.size(0)
        x = x.view(batchsize, 256 * 6 * 6)
        x = self.Fcs(x)
        return x

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self)
        elif pretrained == 'caffemodel':
            None
            src2dsts = dict(conv1='Convs.0', conv2='Convs.4', conv3=
                'Convs.8', conv4='Convs.10', conv5='Convs.12', fc6='Fcs.0',
                fc7='Fcs.3')
            copy_weights(self.state_dict(), 'caffemodel.alexnet', src2dsts=
                src2dsts)
        elif pretrained == 'torchmodel':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def fix_conv1_conv2(self):
        for layer in range(8):
            for p in self.Convs[layer].parameters():
                p.requires_grad = False


def norm2unit(vecs, dim=1, p=2):
    """ vecs is of 2D:  (batchsize x nr_feat) or (batchsize x nr_feat x restDims)
        We normalize it to a unit vector here.
        For example, since mu is the coordinate on the circle (x,y) or sphere (x,y,z),
        we want to make it a unit norm.
    """
    vsize = vecs.size()
    batchsize, nr_feat = vsize[0], vsize[1]
    if hasattr(vecs, 'data'):
        check_inf = torch.abs(vecs.data) == float('inf')
    else:
        check_inf = torch.abs(vecs) == float('inf')
    check_nan = vecs != vecs
    if check_inf.any() or check_nan.any():
        print(vecs)
        print(
            '[Exception] some input values are either "nan" or "inf" !   (from norm2unit)'
            )
        exit()
    signs = torch.sign(vecs)
    signs[signs == 0] = 1
    vecs = vecs + signs * 0.0001
    vecs_in = vecs.clone()
    norm = ((vecs ** p).sum(dim=dim, keepdim=True) ** (1.0 / p)).detach()
    check_bad_norm = torch.abs(norm).lt(0.0001)
    if check_bad_norm.any():
        print(check_bad_norm)
        print('[Exception] some norm close to 0.   (from norm2unit)')
        exit()
    vecs = vecs.div(norm)
    recomputed_norm = torch.sum(vecs ** p, dim=dim)
    check_mu = torch.abs(recomputed_norm - 1.0).lt(1e-06)
    if not check_mu.all():
        np.set_printoptions(threshold=np.inf)
        print(norm.data.cpu().numpy())
        print(torch.cat([vecs_in, vecs, norm, recomputed_norm.view(
            batchsize, 1)], dim=dim)[(~check_mu), :].data.cpu().numpy())
        print('[Exception] normalization has problem.   (from norm2unit)')
        exit()
    return vecs


class Test_AlexNet(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=AlexNet_Trunk):
        super(Test_AlexNet, self).__init__()
        self.truck = _Trunk(init_weights=True)
        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        self.head_s2 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 2))
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3
            ), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2
            ), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


class AlexNet_Trunk(nn.Module):

    def __init__(self):
        super(AlexNet_Trunk, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6,
            4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 
            4096), nn.ReLU(inplace=True))

    def forward_Conv(self, x):
        return self.features(x)

    def forward_Fc(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 227, 227)  """
        x = self.features(x)
        batchsize = x.size(0)
        x = x.view(batchsize, 256 * 6 * 6)
        x = self.classifier(x)
        return x

    @staticmethod
    def init_weights(self):
        copy_weights(self.state_dict(), 'torchmodel.alexnet', strict=False)
        return self


class Test_AlexNet(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=AlexNet_Trunk):
        super(Test_AlexNet, self).__init__()
        self.truck = _Trunk().copy_weights()
        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        self.head_s2 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 2))
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3
            ), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2
            ), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features,
            kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=
            (0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding
            =(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class nnAdd(nn.Module):

    def forward(self, x0, x1):
        return x0 + x1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


def _nn_xNorm(num_channels, xNorm='BN', **kwargs):
    """ E.g.
         Fixed BN:   xNorm='BN', affine=False
    """
    if xNorm == 'BN':
        return nn.BatchNorm2d(num_channels, **kwargs)
    elif xNorm == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=num_channels, **kwargs)
    elif xNorm == 'IN':
        return nn.InstanceNorm2d(num_channels, **kwargs)
    elif xNorm == 'LN':
        raise NotImplementedError


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm=
        'BN', xNorm_kwargs={}):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.downsample = downsample
        self.stride = stride
        self.merge = nnAdd()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.merge(out, residual)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm=
        'BN', xNorm_kwargs={}):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = _nn_xNorm(planes * 4, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.merge = nnAdd()

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
        out = self.merge(out, residual)
        out = self.relu(out)
        return out


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


type2blockFunc_layerCfgs = dict(resnet18=(BasicBlock, [2, 2, 2, 2]),
    resnet34=(BasicBlock, [3, 4, 6, 3]), resnet50=(Bottleneck, [3, 4, 6, 3]
    ), resnet101=(Bottleneck, [3, 4, 23, 3]), resnet152=(Bottleneck, [3, 8,
    36, 3]))


class _ResNet_Trunk(nn.Module):

    def __init__(self, res_type='resnet101', pretrained='torchmodel', start
        =None, end=None, xNorm='BN', xNorm_kwargs={}):
        self.inplanes = 64
        super(_ResNet_Trunk, self).__init__()
        self.net_arch = res_type
        blockFunc, layerCfgs = type2blockFunc_layerCfgs[res_type]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = _nn_xNorm(64, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blockFunc, 64, layerCfgs[0], stride=
            1, xNorm='BN', xNorm_kwargs={})
        self.layer2 = self._make_layer(blockFunc, 128, layerCfgs[1], stride
            =2, xNorm='BN', xNorm_kwargs={})
        self.layer3 = self._make_layer(blockFunc, 256, layerCfgs[2], stride
            =2, xNorm='BN', xNorm_kwargs={})
        self.layer4 = self._make_layer(blockFunc, 512, layerCfgs[3], stride
            =2, xNorm='BN', xNorm_kwargs={})
        self.pool5 = nn.AvgPool2d(7, stride=1)
        self._layer_names = [name for name, module in self.named_children()]
        if pretrained == True:
            pretrained = 'torchmodel'
        if pretrained == False:
            pretrained = None
        assert pretrained in [None, 'caffemodel', 'torchmodel'
            ], 'Unknown pretrained: %s' % pretrained
        None
        self.init_weights(pretrained=pretrained)
        self.truck_seq = self.sub_seq(start=start, end=end)

    def sub_seq(self, start=None, end=None):
        assert start is None or start in self._layer_names, '[Error] %s is not in %s' % (
            start, self._layer_names)
        assert end is None or end in self._layer_names, '[Error] %s is not in %s' % (
            end, self._layer_names)
        start_ind = self._layer_names.index(start) if start is not None else 0
        end_ind = self._layer_names.index(end) if end is not None else len(self
            ._layer_names) - 1
        assert start_ind <= end_ind
        self.selected_layer_name = self._layer_names[start_ind:end_ind + 1]
        None
        _seq = nn.Sequential(*[self.__getattr__(x) for x in self.
            selected_layer_name])
        return _seq

    def _make_layer(self, block, planes, blocks, stride, xNorm, xNorm_kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                _nn_xNorm(planes * block.expansion, xNorm=xNorm, **
                xNorm_kwargs))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            xNorm=xNorm, xNorm_kwargs=xNorm_kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, xNorm=xNorm,
                xNorm_kwargs=xNorm_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        assert x.size()[1:] == (3, 224, 224
            ), 'resnet need (3,224,224) input data, whereas %s is received.' % str(
            tuple(x.size()[1:]))
        if self.selected_layer_name[-1] == 'pool5':
            return self.truck_seq(x).view(x.size(0), -1)
        else:
            return self.truck_seq(x)

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self, silent=False)
        elif pretrained == 'caffemodel':
            model_path = pretrained_cfg.caffemodel.resnet101.model
            None
            state_dict = torch.load(model_path)
            self.load_state_dict({k: v for k, v in state_dict.items() if k in
                self.state_dict()})
        elif pretrained == 'torchmodel':
            None
            state_dict = model_zoo.load_url(model_urls[self.net_arch],
                progress=True)
            self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
        return self


class ResNet101_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet101', **kwargs)


class Test_Net(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=ResNet101_Trunk):
        super(Test_Net, self).__init__()
        self.truck = _Trunk()
        self.nr_cate = nr_cate
        self.head_s2 = nn.Sequential(nn.Linear(2048, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(2048, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 2))
        self.maskout = Maskout(nr_cate=nr_cate)
        init_weights_by_filling(self.head_s2)
        init_weights_by_filling(self.head_s1)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        x = x.view(batchsize, -1)
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3
            ), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2
            ), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_vggm():
    return nn.Sequential(nn.Conv2d(3, 96, (7, 7), (2, 2)), nn.ReLU(inplace=
        True), LocalResponseNorm(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3),
        (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(96, 256, (5, 5), (2, 2),
        (1, 1)), nn.ReLU(inplace=True), LocalResponseNorm(5, 0.0005, 0.75, 
        2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d
        (256, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.
        Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn
        .Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True),
        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True))


def get_caffeSrc2Dst(net_arch='vgg16'):
    if net_arch == 'vgg16':
        return odict(conv1_1='features.0', conv1_2='features.2', conv2_1=
            'features.5', conv2_2='features.7', conv3_1='features.10',
            conv3_2='features.12', conv3_3='features.14', conv4_1=
            'features.17', conv4_2='features.19', conv4_3='features.21',
            conv5_1='features.24', conv5_2='features.26', conv5_3=
            'features.28', fc6='classifier.0', fc7='classifier.3')
    elif net_arch == 'vgg19':
        return odict(conv1_1='features.0', conv1_2='features.2', conv2_1=
            'features.5', conv2_2='features.7', conv3_1='features.10',
            conv3_2='features.12', conv3_3='features.14', conv3_4=
            'features.16', conv4_1='features.19', conv4_2='features.21',
            conv4_3='features.23', conv4_4='features.25', conv5_1=
            'features.28', conv5_2='features.30', conv5_3='features.32',
            conv5_4='features.34', fc6='classifier.0', fc7='classifier.3')
    elif net_arch in ['vgg16_bn', 'vgg19_bn']:
        print('No pretrained model for vgg16_bn, vgg19_bn.')
        raise NotImplementedError
    elif net_arch == 'vggm':
        return odict(conv1='features.0', conv2='features.4', conv3=
            'features.8', conv4='features.10', conv5='features.12', fc6=
            'classifier.0', fc7='classifier.3')
    else:
        raise NotImplementedError


_global_config['D'] = 4


_global_config['E'] = 4


type2cfg = dict(vgg16=cfg['D'], vgg19=cfg['E'])


class _VGG_Trunk(nn.Module):

    def __init__(self, vgg_type='vgg16', init_weights=True):
        super(_VGG_Trunk, self).__init__()
        self.net_arch = vgg_type
        if vgg_type == 'vggm':
            self.features = make_layers_vggm()
            self.classifier = nn.Sequential(nn.Linear(512 * 6 * 6, 4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU
                (True), nn.Dropout())
        else:
            self.features = make_layers(type2cfg[vgg_type])
            self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU
                (True), nn.Dropout())
        if init_weights == True:
            assert self.net_arch in ['vgg16', 'vgg19']
            self.init_weights(pretrained='caffemodel')
        elif isinstance(init_weights, str) or init_weights is None:
            self.init_weights(pretrained=init_weights)
        else:
            raise NotImplementedError
    """
    def sub_seq(self, start=None, end=None):   # 'conv1', 'pool5'
        # select sub-sequence from trunk (by creating a shot cut).
        assert start is None or start in self._layer_names, '[Error] %s is not in %s' % (start, self._layer_names)
        assert end   is None or end   in self._layer_names, '[Error] %s is not in %s' % (end  , self._layer_names)
        start_ind = self._layer_names.index(start) if (start is not None) else 0
        end_ind   = self._layer_names.index(end)   if (end   is not None) else len(self._layer_names)-1
        assert start_ind<=end_ind
        self.selected_layer_name = self._layer_names[start_ind:end_ind+1]
        print("Selected sub-sequence: %s" % self.selected_layer_name)
        _seq = nn.Sequential(*[self.__getattr__(x) for x in self.selected_layer_name])
        return _seq     # (self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,self.layer2,self.layer3 )
    """

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        x = self.features(x)
        batchsize = x.size(0)
        x = x.view(batchsize, -1)
        x = self.classifier(x)
        return x

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self)
        elif pretrained == 'caffemodel':
            None
            src2dsts = get_caffeSrc2Dst(self.net_arch)
            copy_weights(self.state_dict(), 'caffemodel.%s' % self.net_arch,
                src2dsts=src2dsts)
        elif pretrained == 'torchmodel':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def fix_conv1_conv2(self):
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


class Test_Net(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=VGG16_Trunk):
        super(Test_Net, self).__init__()
        self.truck = _Trunk()
        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        self.head_s2 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=
            True), nn.Linear(84, self.nr_cate * 2))
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3
            ), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2
            ), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


class Maskout(nn.Module):

    def __init__(self, nr_cate=3):
        super(Maskout, self).__init__()
        self.nr_cate = nr_cate

    def forward(self, x, label):
        """[input]
             x:    of shape (batchsize, nr_cate, nr_feat)
                or of shape (batchsize, nr_cate)   where nr_feat is treat as 1.
             label:
           [output]
             tensor of shape (batchsize, nr_cate)
        """
        batchsize, _nr_cate = x.size(0), x.size(1)
        assert _nr_cate == self.nr_cate, '2nd dim of x should be self.nr_cate=%s' % self.nr_cate
        assert batchsize == label.size(0)
        assert batchsize == reduce(lambda a1, a2: a1 * a2, label.size())
        item_inds = torch.from_numpy(np.arange(batchsize))
        if x.is_cuda:
            item_inds = item_inds
        cate_ind = label.view(batchsize)
        assert cate_ind.lt(self.nr_cate).all(
            ), '[Exception] All index in cate_ind should be smaller than nr_cate.'
        masked_shape = (batchsize,) + x.size()[2:]
        return x[item_inds, cate_ind].view(*masked_shape)


class LocalResponseNorm(Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        """Applies local response normalization over an input signal composed
        of several input planes, where channels occupy the second dimension.
        Applies normalization across channels.

        .. math::

            `b_{c} = a_{c}\\left(k + \\frac{\\alpha}{n}
            \\sum_{c'=\\max(0, c-n/2)}^{\\min(N-1,c+n/2)}a_{c'}^2\\right)^{-\\beta}`

        Args:
            size: amount of neighbouring channels used for normalization
            alpha: multiplicative factor. Default: 0.0001
            beta: exponent. Default: 0.75
            k: additive factor. Default: 1

        Shape:
            - Input: :math:`(N, C, ...)`
            - Output: :math:`(N, C, ...)` (same shape as input)
        Examples::
            >>> lrn = nn.LocalResponseNorm(2)
            >>> signal_2d = autograd.Variable(torch.randn(32, 5, 24, 24))
            >>> signal_4d = autograd.Variable(torch.randn(16, 5, 7, 7, 7, 7))
            >>> output_2d = lrn(signal_2d)
            >>> output_4d = lrn(signal_4d)
        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta,
            self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.size
            ) + ', alpha=' + str(self.alpha) + ', beta=' + str(self.beta
            ) + ', k=' + str(self.k) + ')'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leoshine_Spherical_Regression(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InceptionA(*[], **{'in_channels': 4, 'pool_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(InceptionB(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(InceptionC(*[], **{'in_channels': 4, 'channels_7x7': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(InceptionD(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(InceptionE(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(LocalResponseNorm(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(UNet(*[], **{'n_channels': 4, 'n_classes': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_009(self):
        self._check(double_conv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(down(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(inconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(nnAdd(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(outconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(up(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

