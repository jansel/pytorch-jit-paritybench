import sys
_module = sys.modules[__name__]
del sys
VGGFeatureLoss = _module
attention = _module
datasets = _module
main = _module
miscc = _module
config = _module
losses = _module
utils = _module
model = _module
pretrain_DAMSM = _module
trainer = _module

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


from torchvision import models


from torchvision import transforms


import torch


import torchvision


import torch.nn as nn


import numpy as np


import torch.utils.model_zoo as model_zoo


from collections import defaultdict


import torch.utils.data as data


from torch.autograd import Variable


import torchvision.transforms as transforms


import pandas as pd


import numpy.random as random


import time


import random


import torch.nn.functional as F


import torchvision.models as models


from torch.nn import init


from copy import deepcopy


import torch.nn.parallel


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.optim as optim


import torch.backends.cudnn as cudnn


class VGGNet(nn.Module):

    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['8']
        model = models.vgg16()
        url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.resquires_grad = False
        None
        self.vgg = model.features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def conv1x1(in_planes, out_planes, bias=False):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


class SpatialAttention(nn.Module):

    def __init__(self, idf, cdf):
        super(SpatialAttention, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.conv_context2 = conv1x1(cdf, 64 * 64)
        self.conv_context3 = conv1x1(cdf, 128 * 128)
        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        weightedContext = torch.bmm(sourceT, attn)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn


class ChannelAttention(nn.Module):

    def __init__(self, idf, cdf):
        super(ChannelAttention, self).__init__()
        self.conv_context2 = conv1x1(cdf, 64 * 64)
        self.conv_context3 = conv1x1(cdf, 128 * 128)
        self.sm = nn.Softmax()
        self.idf = idf

    def forward(self, weightedContext, context, ih, iw):
        batch_size, sourceL = context.size(0), context.size(2)
        sourceC = context.unsqueeze(3)
        if ih == 64:
            sourceC = self.conv_context2(sourceC).squeeze(3)
        else:
            sourceC = self.conv_context3(sourceC).squeeze(3)
        attn_c = torch.bmm(weightedContext, sourceC)
        attn_c = attn_c.view(batch_size * self.idf, sourceL)
        attn_c = self.sm(attn_c)
        attn_c = attn_c.view(batch_size, self.idf, sourceL)
        attn_c = torch.transpose(attn_c, 1, 2).contiguous()
        weightedContext_c = torch.bmm(sourceC, attn_c)
        weightedContext_c = torch.transpose(weightedContext_c, 1, 2).contiguous()
        weightedContext_c = weightedContext_c.view(batch_size, -1, ih, iw)
        return weightedContext_c, attn_c


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


class ResBlock(nn.Module):

    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2), nn.BatchNorm2d(channel_num * 2), GLU(), conv3x3(channel_num, channel_num), nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class RNN_ENCODER(nn.Module):

    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True, dropout=self.drop_prob, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden, self.nlayers, batch_first=True, dropout=self.drop_prob, bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()), Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_())
        else:
            return Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):

    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        None
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        features = x
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        cnn_code = self.emb_cnn_code(x)
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class INIT_STAGE_G(nn.Module):

    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(nn.Linear(nz, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code32 = self.upsample3(out_code)
        out_code64 = self.upsample4(out_code32)
        return out_code64


class NEXT_STAGE_G(nn.Module):

    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = SPATIAL_NET(ngf, self.ef_dim)
        self.channel_att = CHANNEL_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 3)
        self.upsample = upBlock(ngf * 3, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code: batch x idf x queryL
            att: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        c_code_channel, att_channel = self.channel_att(c_code, word_embs, h_code.size(2), h_code.size(3))
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        out_code = self.residual(h_c_c_code)
        out_code = self.upsample(out_code)
        return out_code, att


class GET_IMAGE_G(nn.Module):

    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):

    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        return fake_imgs, att_maps, mu, logvar


class G_DCGAN(nn.Module):

    def __init__(self):
        super(G_DCGAN, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
        self.img_net = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
        """
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code = self.h_net1(z_code, c_code)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code, att1 = self.h_net2(h_code, c_code, word_embs, mask)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code, att2 = self.h_net3(h_code, c_code, word_embs, mask)
            if att2 is not None:
                att_maps.append(att2)
        fake_imgs = self.img_net(h_code)
        return [fake_imgs], att_maps, mu, logvar


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))
    return block


class D_GET_LOGITS(nn.Module):

    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)
        self.outlogits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)


def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
    return encode_img


class D_NET64(nn.Module):

    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code_s16(x_var)
        return x_code4


def downBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))
    return block


class D_NET128(nn.Module):

    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)
        x_code4 = self.img_code_s32(x_code8)
        x_code4 = self.img_code_s32_1(x_code4)
        return x_code4


class D_NET256(nn.Module):

    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (D_GET_LOGITS,
     lambda: ([], {'ndf': 4, 'nef': 4}),
     lambda: ([torch.rand([4, 32, 64, 64])], {}),
     False),
    (GET_IMAGE_G,
     lambda: ([], {'ngf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'channel_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialAttention,
     lambda: ([], {'idf': 4, 'cdf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (VGGNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_mrlibw_ControlGAN(_paritybench_base):
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

