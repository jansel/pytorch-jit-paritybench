import sys
_module = sys.modules[__name__]
del sys
master = _module
checkpoint = _module
DHF1K = _module
HMDB51 = _module
ImageNetVID = _module
KTH = _module
MSCOCO = _module
Manual_Hands = _module
UCF101 = _module
VOC2007 = _module
YC2BB = _module
datasets = _module
abstract_datasets = _module
loading_function = _module
preprocessing_transforms = _module
gen_json_DHF1K = _module
gen_json_HMDB51 = _module
gen_json_imagenetvid = _module
gen_json_mscoco = _module
gen_json_voc2007 = _module
gen_json_yc2bb = _module
eval = _module
losses = _module
metrics = _module
models = _module
c3d = _module
dvsa = _module
transformer = _module
i3d = _module
models_import = _module
ssd = _module
ssd_utils = _module
box_utils = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
parse_args = _module
train = _module

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


import numpy as np


import torch.nn as nn


import torch.optim as optim


import torch.utils.data as Data


from scipy import ndimage


import torch.nn.functional as F


import math


from functools import partial


from torch import nn


from torch.nn import functional as F


from torch.autograd import Variable


import random


import string


import uuid


from collections import OrderedDict


from torch.autograd import Function


import torch.nn.init as init


from torch.optim.lr_scheduler import MultiStepLR


class PreprocessTrainC3D(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """
        self.transforms = []
        self.transforms1 = []
        self.preprocess = kwargs['preprocess']
        crop_type = kwargs['crop_type']
        self.clip_mean = np.load('weights/sport1m_train16_128_mean.npy')[0]
        self.clip_mean = np.transpose(self.clip_mean, (1, 2, 3, 0))
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.SubtractMeanClip(clip_mean=self.clip_mean,
            **kwargs))
        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        else:
            self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.RandomFlipClip(direction='h', p=0.5, **
            kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)
        return input_data


class PreprocessEvalC3D(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """
        self.transforms = []
        self.clip_mean = np.load('weights/sport1m_train16_128_mean.npy')[0]
        self.clip_mean = np.transpose(self.clip_mean, (1, 2, 3, 0))
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.SubtractMeanClip(clip_mean=self.clip_mean,
            **kwargs))
        self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)
        return input_data


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, **kwargs):
        """
        Initialize C3D model  
        Args:
            labels     (Int):    Total number of classes in the dataset
            pretrained (Int/String): Initialize with random (0) or pretrained (1) weights 

        Return:
            None
        """
        super(C3D, self).__init__()
        self.train_transforms = PreprocessTrainC3D(**kwargs)
        self.test_transforms = PreprocessEvalC3D(**kwargs)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 
            1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
            padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, kwargs['labels'])
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()
        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            self.__load_pretrained_weights()

    def forward(self, x, labels=False):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        if labels:
            logits = F.softmax(logits, dim=1)
        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {'features.0.weight': 'conv1.weight',
            'features.0.bias': 'conv1.bias', 'features.3.weight':
            'conv2.weight', 'features.3.bias': 'conv2.bias',
            'features.6.weight': 'conv3a.weight', 'features.6.bias':
            'conv3a.bias', 'features.8.weight': 'conv3b.weight',
            'features.8.bias': 'conv3b.bias', 'features.11.weight':
            'conv4a.weight', 'features.11.bias': 'conv4a.bias',
            'features.13.weight': 'conv4b.weight', 'features.13.bias':
            'conv4b.bias', 'features.16.weight': 'conv5a.weight',
            'features.16.bias': 'conv5a.bias', 'features.18.weight':
            'conv5b.weight', 'features.18.bias': 'conv5b.bias',
            'classifier.0.weight': 'fc6.weight', 'classifier.0.bias':
            'fc6.bias', 'classifier.3.weight': 'fc7.weight',
            'classifier.3.bias': 'fc7.bias'}
        p_dict = torch.load('weights/c3d-pretrained.pth')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DVSA(nn.Module):
    """
    Deep Visual-Semantic Alignments (DVSA). 
    Implementation used as baseline in Weakly-Supervised Video Object Grounding...
    Source: https://arxiv.org/pdf/1805.02834.pdf
    
    Original paper: Deep visual-semantic alignments for generating image descriptions
    https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
    """

    def __init__(self, **kwargs):
        super().__init__()
        num_class = kwargs['labels']
        input_size = kwargs['input_size']
        enc_size = kwargs['enc_size']
        dropout = kwargs['dropout']
        hidden_size = kwargs['hidden_size']
        n_layers = kwargs['n_layers']
        n_heads = kwargs['n_heads']
        attn_drop = kwargs['attn_drop']
        num_frm = kwargs['yc2bb_num_frm']
        has_loss_weighting = kwargs['has_loss_weighting']
        self.feat_enc = nn.Sequential(nn.Linear(input_size, enc_size), nn.
            Dropout(p=dropout), nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.obj_emb = nn.Embedding(num_class + 1, enc_size)
        self.num_class = num_class
        self.obj_interact = Transformer(enc_size, 0, 0, d_hidden=
            hidden_size, n_layers=n_layers, n_heads=n_heads, drop_ratio=
            attn_drop)
        self.obj_interact_fc = nn.Sequential(nn.Linear(enc_size * 2, int(
            enc_size / 2)), nn.ReLU(), nn.Linear(int(enc_size / 2), 5), nn.
            Sigmoid())
        self.num_frm = num_frm
        self.has_loss_weighting = has_loss_weighting
        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            self._load_pretrained_weights()

    def forward(self, x_o, obj, load_type):
        is_evaluate = 1 if load_type[0] == 'test' or load_type[0
            ] == 'val' else 0
        if is_evaluate:
            return self.output_attn(x_o, obj)
        x_o = x_o[0]
        obj = obj[0]
        x_o = self.feat_enc(x_o.permute(0, 2, 3, 1).contiguous()).permute(0,
            3, 1, 2).contiguous()
        x_o = torch.stack([x_o[0], x_o[1], x_o[0]])
        obj = torch.stack([obj[0], obj[0], obj[1]])
        N, C_out, T, num_proposals = x_o.size()
        assert N == 3
        O = obj.size(1)
        attn_key = self.obj_emb(obj)
        num_pos_obj = torch.sum(obj[0] < self.num_class).long().item()
        num_neg_obj = torch.sum(obj[2] < self.num_class).long().item()
        attn_key_frm_feat = attn_key[0:1, :num_pos_obj]
        obj_attn_emb, _ = self.obj_interact(attn_key_frm_feat)
        obj_attn_emb = obj_attn_emb[:, :num_pos_obj, :]
        obj_attn_emb = torch.cat((obj_attn_emb, attn_key[0:1, :num_pos_obj]
            ), dim=2)
        obj_attn_emb = self.obj_interact_fc(obj_attn_emb)
        itv = math.ceil(T / 5)
        tmp = []
        for i in range(5):
            l = min(itv * (i + 1), T) - itv * i
            if l > 0:
                tmp.append(obj_attn_emb[:, :, i:i + 1].expand(1,
                    num_pos_obj, l))
        obj_attn_emb = torch.cat(tmp, 2).squeeze(0)
        assert obj_attn_emb.size(1) == self.num_frm
        loss_weigh = torch.mean(obj_attn_emb, dim=0)
        loss_weigh = torch.cat((loss_weigh, loss_weigh)).unsqueeze(1)
        if self.has_loss_weighting:
            x_o = x_o.view(N, 1, C_out, T, num_proposals)
            attn_weights = self.sigmoid((x_o * attn_key.view(N, O, C_out, 1,
                1)).sum(2) / math.sqrt(C_out))
            pos_weights = attn_weights[(0), :num_pos_obj, :, :]
            neg1_weights = attn_weights[(1), :num_pos_obj, :, :]
            neg2_weights = attn_weights[(2), :num_neg_obj, :, :]
            return torch.cat((torch.stack((torch.mean(torch.max(pos_weights,
                dim=2)[0], dim=0), torch.mean(torch.max(neg1_weights, dim=2
                )[0], dim=0)), dim=1), torch.stack((torch.mean(torch.max(
                pos_weights, dim=2)[0], dim=0), torch.mean(torch.max(
                neg2_weights, dim=2)[0], dim=0)), dim=1))), loss_weigh
        else:
            x_o = x_o.view(N, 1, C_out, T * num_proposals)
            attn_weights = self.sigmoid((x_o * attn_key.view(N, O, C_out, 1
                )).sum(2) / math.sqrt(C_out))
            pos_weights = attn_weights[(0), :num_pos_obj, :]
            neg1_weights = attn_weights[(1), :num_pos_obj, :]
            neg2_weights = attn_weights[(2), :num_neg_obj, :]
            return torch.stack((torch.stack((torch.mean(torch.max(
                pos_weights, dim=1)[0]), torch.mean(torch.max(neg1_weights,
                dim=1)[0]))), torch.stack((torch.mean(torch.max(pos_weights,
                dim=1)[0]), torch.mean(torch.max(neg2_weights, dim=1)[0]))))
                ), loss_weigh

    def output_attn(self, x_o, obj):
        x_o = self.feat_enc(x_o.permute(0, 2, 3, 1).contiguous()).permute(0,
            3, 1, 2).contiguous()
        N, C_out, T, num_proposals = x_o.size()
        assert N == 1
        O = obj.size(1)
        attn_key = self.obj_emb(obj)
        x_o = x_o.view(N, 1, C_out, T * num_proposals)
        attn_weights = self.sigmoid((x_o * attn_key.view(N, O, C_out, 1)).
            sum(2) / math.sqrt(C_out))
        return attn_weights.view(N, O, T, num_proposals)

    def _load_pretrained_weights(self):
        state_dict = torch.load('weights/yc2bb_full-model.pth',
            map_location=lambda storage, location: storage)
        self.load_state_dict(state_dict)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*
            size[:-1], -1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


INF = 10000000000.0


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri
            dot_products.data.sub_(tri.unsqueeze(0))
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim
            =2)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.wo = Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.n_heads, -1) for x in (query, key,
            value))
        return self.wo(torch.cat([self.attention(q, k, v) for q, k, v in
            zip(query, key, value)], -1))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(MultiHead(d_model, d_model, n_heads,
            drop_ratio), d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
            d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = x.new(*x.size()[1:]).fill_(0)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, (channel)] = torch.sin(positions.float() / 10000 **
                (channel / x.size(2)))
        else:
            encodings[:, (channel)] = torch.cos(positions.float() / 10000 **
                ((channel - 1) / x.size(2)))
    return Variable(encodings)


class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
        drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_hidden,
            n_heads, drop_ratio) for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        x = x + positional_encodings_like(x)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
            encoding.append(x)
        return encoding


class Transformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
        n_layers=6, n_heads=8, drop_ratio=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
            n_heads, drop_ratio)

    def denum(self, data):
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '#').replace(' <pad>', '')

    def forward(self, x):
        encoding = self.encoder(x)
        return encoding[-1], encoding


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
        stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=
        True, use_bias=False, name='unit_3d', dilation=1):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self.
            _output_channels, kernel_size=self._kernel_shape, stride=self.
            _stride, padding=0, bias=self._use_bias, dilation=dilation)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001,
                momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=
            out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name +
            '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=
            out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name +
            '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=
            out_channels[2], kernel_shape=[3, 3, 3], name=name +
            '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=
            out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name +
            '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=
            out_channels[4], kernel_shape=[3, 3, 3], name=name +
            '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1,
            1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=
            out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name +
            '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class PreprocessTrain(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """
        self.transforms = []
        self.transforms1 = []
        self.preprocess = kwargs['preprocess']
        crop_type = kwargs['crop_type']
        self.transforms.append(pt.ResizeClip(**kwargs))
        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        else:
            self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.RandomFlipClip(direction='h', p=0.5, **
            kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)
        return input_data


class PreprocessEval(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """
        self.transforms = []
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)
        return input_data


class I3D(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """
    VALID_ENDPOINTS = ('Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1',
        'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c',
        'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
        'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits',
        'Predictions')

    def __init__(self, spatial_squeeze=True, final_endpoint='Logits', name=
        'inception_i3d', in_channels=3, dropout_keep_prob=0.5, **kwargs):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        super(I3D, self).__init__()
        self._num_classes = kwargs['labels']
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.train_transforms = PreprocessTrain(**kwargs)
        self.test_transforms = PreprocessEval(**kwargs)
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint
                )
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels,
            output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2),
            padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3,
            3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels
            =64, kernel_shape=[1, 1, 1], padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels
            =192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3,
            3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16,
            32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 
            32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3,
            3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [
            192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [
            160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [
            128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [
            112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [
            256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2,
            2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
            [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
            [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes, kernel_shape=[1, 1, 1],
            padding=0, activation_fn=None, use_batch_norm=False, use_bias=
            True, name='logits')
        self.build()
        if 'pretrained' in kwargs.keys() and kwargs['pretrained']:
            if 'i3d_pretrained' in kwargs.keys():
                self._load_checkpoint(kwargs['i3d_pretrained'])
            else:
                self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        p_dict = torch.load('weights/i3d_rgb_imagenet.pt')
        s_dict = self.state_dict()
        for name in p_dict:
            if name in s_dict.keys():
                if p_dict[name].shape == s_dict[name].shape:
                    s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def _load_checkpoint(self, saved_weights):
        p_dict = torch.load(saved_weights)['state_dict']
        s_dict = self.state_dict()
        for name in p_dict:
            if name in s_dict.keys():
                if p_dict[name].shape == s_dict[name].shape:
                    s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes, kernel_shape=[1, 1, 1],
            padding=0, activation_fn=None, use_batch_norm=False, use_bias=
            True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        logits = torch.mean(logits, dim=2)
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    x2 = boxes[:, (2)]
    y2 = boxes[:, (3)]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:,
        2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes
            ).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[(i), (cl), :count] = torch.cat((scores[ids[:count]].
                    unsqueeze(1), boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, (0)].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PreprocessTrainSSD(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """

    def __init__(self, **kwargs):
        crop_shape = kwargs['crop_shape']
        crop_type = kwargs['crop_type']
        resize_shape = kwargs['resize_shape']
        self.transforms = []
        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.ToTensorClip())

    def __call__(self, input_data, bbox_data=[]):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """
        if bbox_data == []:
            for transform in self.transforms:
                input_data = transform(input_data)
            return input_data
        else:
            for transform in self.transforms:
                input_data, bbox_data = transform(input_data, bbox_data)
            return input_data, bbox_data


class PreprocessEvalSSD(object):
    """
    Container for all transforms used to preprocess clips for evaluation in this dataset.
    """

    def __init__(self, **kwargs):
        crop_shape = kwargs['crop_shape']
        crop_type = kwargs['crop_type']
        resize_shape = kwargs['resize_shape']
        self.transforms = []
        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(**kwargs))
        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.SubtractRGBMean(**kwargs))
        self.transforms.append(pt.ToTensorClip())

    def __call__(self, input_data, bbox_data=[]):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """
        if bbox_data == []:
            for transform in self.transforms:
                input_data = transform(input_data)
            return input_data
        else:
            for transform in self.transforms:
                input_data, bbox_data = transform(input_data, bbox_data)
            return input_data, bbox_data


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(
                    1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=
        True)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4,
            kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes,
            kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3,
            padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes,
            kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        load_type: (string) Can be "test" or "train"
        resize_shape: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, **kwargs):
        super(SSD, self).__init__()
        self.train_transforms = PreprocessTrainSSD(**kwargs)
        self.test_transforms = PreprocessEvalSSD(**kwargs)
        self.load_type = kwargs['load_type']
        self.num_classes = kwargs['labels']
        self.cfg = {'num_classes': 21, 'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000, 'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes':
            [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 
            264, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2],
            [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'VOC'}
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = kwargs['resize_shape'][0]
        base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512,
            512, 512, 'M', 512, 512, 512], '512': []}
        extras = {'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
            '512': []}
        mbox = {'300': [4, 6, 6, 6, 4, 4], '512': []}
        base, extras, head = multibox(vgg(base[str(self.size)], 3),
            add_extras(extras[str(self.size)], 1024), mbox[str(self.size)],
            self.num_classes)
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.load_type == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on load_type:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        if len(x.shape) > 4:
            x.squeeze_(2)
        assert len(x.shape) == 4
        sources = list()
        loc = list()
        conf = list()
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.load_type == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax
                (conf.view(conf.size(0), -1, self.num_classes)), self.
                priors.to(x.device))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes), self.priors
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file, map_location=lambda
                storage, loc: storage))
            None
        else:
            None


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x
            ) * x
        return out


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 
        2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:,
        :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, (0)] * inter[:, :, (1)]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])
        ).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])
        ).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes
        [:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
        bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu
        =True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view
            (-1, 1))
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes
            )
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MichiganCOG_ViP(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Linear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(LayerNorm(*[], **{'d_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FeedForward(*[], **{'d_model': 4, 'd_hidden': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Transformer(*[], **{'d_model': 4, 'n_vocab_src': 4, 'vocab_trg': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Unit3D(*[], **{'in_channels': 4, 'output_channels': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    def test_005(self):
        self._check(L2Norm(*[], **{'n_channels': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

