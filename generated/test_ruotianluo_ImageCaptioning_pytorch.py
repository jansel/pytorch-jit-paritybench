import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
dataloaderraw = _module
eval = _module
eval_utils = _module
misc = _module
resnet = _module
resnet_utils = _module
utils = _module
Att2inModel = _module
AttModel = _module
CaptionModel = _module
FCModel = _module
OldModel = _module
ShowTellModel = _module
models = _module
opts = _module
convert_old = _module
prepro_feats = _module
prepro_labels = _module
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


import numpy as np


import random


import torch.utils.data as data


import torch


from torch.autograd import Variable


import scipy.misc


from torchvision import transforms as trn


import time


import torch.nn as nn


import string


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import collections


from torch.autograd import *


import copy


from random import shuffle


from random import seed


import torchvision.models as models


import torch.optim as optim


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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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


class myResnet(nn.Module):

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)
        return fc, att


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]
        input = to_contiguous(input).view(-1, input.shape[2])
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class Att2inCore(nn.Module):

    def __init__(self, opt):
        super(Att2inCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(state[0][-1])
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot)
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class AdaAtt_lstm(nn.Module):

    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_maxout = use_maxout
        self.w2h = nn.Linear(self.input_encoding_size, (4 + (use_maxout == True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size)
        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers)])
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, xt, img_fc, state):
        hs = []
        cs = []
        for L in range(self.num_layers):
            prev_h = state[0][L]
            prev_c = state[1][L]
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L - 1](x)
            all_input_sums = i2h + self.h2h[L](prev_h)
            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
            next_c = forget_gate * prev_c + in_gate * in_transform
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers - 1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h + self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c
            cs.append(next_c)
            hs.append(next_h)
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)
        state = torch.cat([_.unsqueeze(0) for _ in hs], 0), torch.cat([_.unsqueeze(0) for _ in cs], 0)
        return top_h, fake_region, state


class AdaAtt_attention(nn.Module):

    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.fr_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.ho_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.Tanh(), nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)
        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)
        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))
        img_all = torch.cat([fake_region.view(-1, 1, self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.input_encoding_size), conv_feat_embed], 1)
        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA, self.drop_prob_lm, self.training)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1))
        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)
        atten_out = visAttdim + h_out_linear
        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class AdaAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats)
        return atten_out, state


class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


class TopDownCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats)
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])
        return output, state


class Att2in2Core(nn.Module):

    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size), in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class CaptionModel(nn.Module):

    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, logprobs, *args, **kwargs):

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q, p=candidate_logprob, r=local_logprob))
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, (vix)] = beam_seq_prev[:, (v['q'])]
                    beam_seq_logprobs[:t, (vix)] = beam_seq_logprobs_prev[:, (v['q'])]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, (vix)] = state[state_ix][:, (v['q'])]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []
        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()
            logprobsf[:, (logprobsf.size(1) - 1)] = logprobsf[:, (logprobsf.size(1) - 1)] - 1000
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates_divm = beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state)
            for vix in range(beam_size):
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {'seq': beam_seq[:, (vix)].clone(), 'logps': beam_seq_logprobs[:, (vix)].clone(), 'p': beam_logprobs_sum[vix]}
                    done_beams.append(final_beam)
                    beam_logprobs_sum[vix] = -1000
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it), *(args + (state,)))
        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams


class LSTMCore(nn.Module):

    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = torch.max(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size), all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        next_h = self.dropout(next_h)
        output = next_h
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class FCModel(CaptionModel):

    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.ss_prob = 0.0
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()), Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, (i - 1)].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, (i - 1)].data.clone()
                        prob_prev = torch.exp(outputs[-1].data)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, (i - 1)].clone()
                if i >= 2 and seq[:, (i - 1)].data.sum() == 0:
                    break
                xt = self.embed(it)
            output, state = self.core(xt, state)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        xt = self.embed(it)
        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output))
        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output))
            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, (k)] = self.done_beams[k][0]['seq']
            seqLogprobs[:, (k)] = self.done_beams[k][0]['logps']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                    it = it.view(-1).long()
                xt = self.embed(Variable(it, requires_grad=False))
            if t >= 2:
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))
            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class OldModel(CaptionModel):

    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.ss_prob = 0.0
        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return image_map, image_map
        else:
            return image_map

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)
        outputs = []
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, (i)].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, (i)].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, (i)].clone()
            if i >= 1 and seq[:, (i)].data.sum() == 0:
                break
            xt = self.embed(it)
            output, state = self.core(xt, fc_feats, att_feats, state)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        xt = self.embed(it)
        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))
        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            state = self.init_hidden(tmp_fc_feats)
            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []
            for t in range(1):
                if t == 0:
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, opt=opt)
            seq[:, (k)] = self.done_beams[k][0]['seq']
            seqLogprobs[:, (k)] = self.done_beams[k][0]['logps']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                it = it.view(-1).long()
            xt = self.embed(Variable(it, requires_grad=False))
            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))
            output, state = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class ShowAttendTellCore(nn.Module):

    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)
            att = att.view(-1, att_size, self.att_hid_size)
            att_h = self.h2att(state[0][-1])
            att_h = att_h.unsqueeze(1).expand_as(att)
            dot = att + att_h
            dot = F.tanh(dot)
            dot = dot.view(-1, self.att_hid_size)
            dot = self.alpha_net(dot)
            dot = dot.view(-1, att_size)
        else:
            att = self.ctx2att(att)(att)
            att = att.view(-1, att_size)
            att_h = self.h2att(state[0][-1])
            att_h = att_h.expand_as(att)
            dot = att_h + att
        weight = F.softmax(dot)
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
        return output.squeeze(0), state


class AllImgCore(nn.Module):

    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, state):
        output, state = self.rnn(torch.cat([xt, fc_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state


class ShowAttendTellModel(OldModel):

    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)


class AllImgModel(OldModel):

    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)


class ShowTellModel(CaptionModel):

    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.ss_prob = 0.0
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()), Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, (i - 1)].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, (i - 1)].data.clone()
                        prob_prev = torch.exp(outputs[-1].data)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, (i - 1)].clone()
                if i >= 2 and seq[:, (i - 1)].data.sum() == 0:
                    break
                xt = self.embed(it)
            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        xt = self.embed(it)
        output, state = self.core(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, (k)] = self.done_beams[k][0]['seq']
            seqLogprobs[:, (k)] = self.done_beams[k][0]['logps']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                    it = it.view(-1).long()
                xt = self.embed(Variable(it, requires_grad=False))
            if t >= 2:
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))
            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaAtt_attention,
     lambda: ([], {'opt': _mock_config(input_encoding_size=4, rnn_size=4, drop_prob_lm=0.5, att_hid_size=4)}),
     lambda: ([torch.rand([1024, 4]), torch.rand([64, 4, 4, 4]), torch.rand([1024, 4, 4, 4]), torch.rand([1024, 4, 4, 4])], {}),
     True),
    (Attention,
     lambda: ([], {'opt': _mock_config(rnn_size=4, att_hid_size=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShowAttendTellCore,
     lambda: ([], {'opt': _mock_config(input_encoding_size=4, rnn_type='gru', rnn_size=4, num_layers=1, drop_prob_lm=0.5, fc_feat_size=4, att_feat_size=4, att_hid_size=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([1, 4, 4])], {}),
     False),
]

class Test_ruotianluo_ImageCaptioning_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

