import sys
_module = sys.modules[__name__]
del sys
arguments = _module
eval = _module
eval_ensemble = _module
lib = _module
_init_paths = _module
datasets = _module
image_caption = _module
encoders = _module
evaluation = _module
loss = _module
modules = _module
aggr = _module
gpo = _module
mlp = _module
resnet = _module
vse = _module
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


import torch


import torch.utils.data as data


import numpy as np


import random


import logging


import torch.nn as nn


from collections import OrderedDict


import time


from torch.autograd import Variable


import math


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import torch.nn.init


import torch.backends.cudnn as cudnn


from torch.nn.utils import clip_grad_norm_


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError('Cannot use sin/cos positional encoding with odd dim (got dim={:d})'.format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class GPO(nn.Module):

    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden
        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)
        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000
        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)
        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)
        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


def l2norm(X, dim, eps=1e-08):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImageAggr(nn.Module):

    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            features = self.mlp(images) + features
        features, pool_weights = self.gpool(features, image_lengths)
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        return features


logger = logging.getLogger(__name__)


class EncoderImageFull(nn.Module):

    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        if self.training:
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.2 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1))
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths)
        else:
            feat_lengths = torch.zeros(base_features.size(0))
            feat_lengths[:] = base_features.size(1)
        features = self.image_encoder(base_features, feat_lengths)
        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


class EncoderText(nn.Module):

    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]
        cap_len = lengths
        cap_emb = self.linear(bert_emb)
        pooled_features, pool_weights = self.gpool(cap_emb, cap_len)
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)
        return pooled_features


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        None

    def max_violation_off(self):
        self.max_violation = False
        None

    def forward(self, im, s):
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class TwoLayerMLP(nn.Module):

    def __init__(self, num_features, hid_dim, out_dim, return_hidden=False):
        super().__init__()
        self.return_hidden = return_hidden
        self.model = nn.Sequential(nn.Linear(num_features, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim))
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not self.return_hidden:
            return self.model(x)
        else:
            hid_feat = self.model[:2](x)
            results = self.model[2:](hid_feat)
            return hid_feat, results


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, width_mult, num_classes=1000):
        self.inplanes = 64 * width_mult
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * width_mult, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width_mult, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width_mult, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion * width_mult, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model_urls = {'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', 'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth'}


def resnet101(pretrained=False, width_mult=1):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, width_mult=1):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet50(pretrained=False, width_mult=1):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class ResnetFeatureExtractor(nn.Module):

    def __init__(self, backbone_source, weights_path, pooling_size=7, fixed_blocks=2):
        super(ResnetFeatureExtractor, self).__init__()
        self.backbone_source = backbone_source
        self.weights_path = weights_path
        self.pooling_size = pooling_size
        self.fixed_blocks = fixed_blocks
        if 'detector' in self.backbone_source:
            self.resnet = resnet101()
        elif self.backbone_source == 'imagenet':
            self.resnet = resnet101(pretrained=True)
        elif self.backbone_source == 'imagenet_res50':
            self.resnet = resnet50(pretrained=True)
        elif self.backbone_source == 'imagenet_res152':
            self.resnet = resnet152(pretrained=True)
        elif self.backbone_source == 'imagenet_resnext':
            self.resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnext101_32x8d', pretrained=True)
        elif 'wsl' in self.backbone_source:
            self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        else:
            raise ValueError('Unknown backbone source {}'.format(self.backbone_source))
        self._init_modules()

    def _init_modules(self):
        self.base = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1, self.resnet.layer2, self.resnet.layer3)
        self.top = nn.Sequential(self.resnet.layer4)
        if self.weights_path != '':
            if 'detector' in self.backbone_source:
                if os.path.exists(self.weights_path):
                    logger.info('Loading pretrained backbone weights from {} for backbone source {}'.format(self.weights_path, self.backbone_source))
                    backbone_ckpt = torch.load(self.weights_path)
                    self.base.load_state_dict(backbone_ckpt['base'])
                    self.top.load_state_dict(backbone_ckpt['top'])
                else:
                    raise ValueError('Could not find weights for backbone CNN at {}'.format(self.weights_path))
            else:
                logger.info('Did not load external checkpoints')
        self.unfreeze_base()

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def unfreeze_base(self):
        assert 0 <= self.fixed_blocks < 4
        if self.fixed_blocks == 3:
            for p in self.base[6].parameters():
                p.requires_grad = False
            for p in self.base[5].parameters():
                p.requires_grad = False
            for p in self.base[4].parameters():
                p.requires_grad = False
            for p in self.base[0].parameters():
                p.requires_grad = False
            for p in self.base[1].parameters():
                p.requires_grad = False
        if self.fixed_blocks == 2:
            for p in self.base[6].parameters():
                p.requires_grad = True
            for p in self.base[5].parameters():
                p.requires_grad = False
            for p in self.base[4].parameters():
                p.requires_grad = False
            for p in self.base[0].parameters():
                p.requires_grad = False
            for p in self.base[1].parameters():
                p.requires_grad = False
        if self.fixed_blocks == 1:
            for p in self.base[6].parameters():
                p.requires_grad = True
            for p in self.base[5].parameters():
                p.requires_grad = True
            for p in self.base[4].parameters():
                p.requires_grad = False
            for p in self.base[0].parameters():
                p.requires_grad = False
            for p in self.base[1].parameters():
                p.requires_grad = False
        if self.fixed_blocks == 0:
            for p in self.base[6].parameters():
                p.requires_grad = True
            for p in self.base[5].parameters():
                p.requires_grad = True
            for p in self.base[4].parameters():
                p.requires_grad = True
            for p in self.base[0].parameters():
                p.requires_grad = True
            for p in self.base[1].parameters():
                p.requires_grad = True
        logger.info('Resnet backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.base.apply(set_bn_eval)
            self.top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.top(pool5).mean(3).mean(2)
        return fc7

    def forward(self, im_data):
        b_s = im_data.size(0)
        base_feat = self.base(im_data)
        top_feat = self.top(base_feat)
        features = top_feat.view(b_s, top_feat.size(1), -1).permute(0, 2, 1)
        return features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContrastiveLoss,
     lambda: ([], {'opt': _mock_config()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (EncoderImageFull,
     lambda: ([], {'backbone_cnn': _mock_layer(), 'img_dim': 4, 'embed_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (TwoLayerMLP,
     lambda: ([], {'num_features': 4, 'hid_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_woodfrog_vse_infty(_paritybench_base):
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

