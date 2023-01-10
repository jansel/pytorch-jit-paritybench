import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
libs = _module
omninet = _module
base_models = _module
resnet = _module
Layers = _module
SubLayers = _module
cnp = _module
cnp = _module
omninet = _module
peripherals = _module
routines = _module
util = _module
utils = _module
bleu = _module
cocoapi = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
coco = _module
eval = _module
meteor = _module
rouge = _module
ptbtokenizer = _module
dataloaders = _module
train_util = _module
vqa = _module
vqaEval = _module
predict = _module
execute_notebook = _module
init_setup = _module
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


import time


import numpy as np


import torch.multiprocessing as mp


import torch.nn as nn


import torch.nn.functional as F


from torch.optim.adam import Adam


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torch.utils.model_zoo as model_zoo


from torch.nn.functional import log_softmax


from torch.nn.functional import softmax


from torch.nn.functional import relu


import torch as T


from torch.autograd import Variable as var


from torch.autograd import Variable


from sklearn.model_selection import train_test_split


from torchvision.datasets import ImageFolder


import matplotlib


import warnings


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
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, k_gate=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if k_gate is not None:
            attn = torch.mul(attn, k_gate)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, k_gate=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        if k_gate is not None:
            k_gate = k_gate.transpose(0, 1)
            k_gate = k_gate.reshape(n_head * sz_b, len_q, len_v)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask, k_gate=k_gate)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        attn = attn.view(n_head, sz_b, len_q, len_v).transpose(0, 1)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, temporal_dim, spatial_dim, dropout=0.1, gpu_id=-1):
        super(DecoderLayer, self).__init__()
        self.gpu_id = gpu_id
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.temporal_cache_attn = MultiHeadAttention(n_head, temporal_dim, d_k, d_v, dropout=dropout)
        self.temporal_proj = nn.Linear(d_model, temporal_dim)
        self.spatial_proj = nn.Linear(temporal_dim, spatial_dim)
        self.spatial_cache_attn = MultiHeadAttention(n_head, spatial_dim, d_k, d_v, dropout=dropout)
        self.spat_dec_proj = nn.Linear(spatial_dim, d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, temporal_cache, spatial_cache, temporal_spatial_link, non_pad_mask, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask
        dec_temp = self.temporal_proj(dec_output)
        dec_temp, dec_temp_attn = self.temporal_cache_attn(dec_temp, temporal_cache, temporal_cache, mask=dec_enc_attn_mask)
        if non_pad_mask is not None:
            dec_temp *= non_pad_mask
        dec_spat = self.spatial_proj(dec_temp)
        dec_spat_attn = None
        if spatial_cache is not None:
            spatial_gate = []
            idx_start = 0
            for l in temporal_spatial_link:
                t, s = l
                if s > 1:
                    temp_sel = dec_temp_attn[:, :, :, idx_start:idx_start + t]
                    b, nh, dq, t = temp_sel.shape
                    temp_sel = temp_sel.unsqueeze(4).expand(b, nh, dq, t, s).transpose(3, 4)
                    temp_sel = temp_sel.reshape(b, nh, dq, t * s)
                    spatial_gate.append(temp_sel)
                idx_start = idx_start + t
            spatial_gate = torch.cat(spatial_gate, dim=3)
            dec_spat, dec_spat_attn = self.spatial_cache_attn(dec_spat, spatial_cache, spatial_cache, k_gate=spatial_gate)
            if non_pad_mask is not None:
                dec_spat *= non_pad_mask
        dec_output = self.spat_dec_proj(dec_spat)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None:
            dec_output *= non_pad_mask
        return dec_output, [dec_slf_attn, dec_spat_attn, dec_temp_attn]


class ControlPeripheral(nn.Module):
    """
        A special peripheral used to help the CNP identify the data domain or specify the context of
        the current operation.

    """

    def __init__(self, control_dim, control_states, gpu_id=-1):
        """
            Accepts as input control states as list of string. The control states are sorted before id's
            are assigned
        """
        super(ControlPeripheral, self).__init__()
        self.control_dim = control_dim
        self.gpu_id = gpu_id
        self.control_dict = {}
        for i, c in enumerate(control_states):
            self.control_dict[c] = i
        self.control_embeddings = nn.Embedding(len(control_states) + 1, self.control_dim)

    def forward(self, control_state, shape=()):
        if self.gpu_id >= 0:
            control_ids = torch.ones(shape, dtype=torch.long, device=self.gpu_id) * self.control_dict[control_state]
        else:
            control_ids = torch.ones(shape, dtype=torch.long) * self.control_dict[control_state]
        return self.control_embeddings(control_ids)


def get_attn_key_pad_mask(pad_mask, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask


def get_non_pad_mask(seq, pad_mask):
    if pad_mask is None:
        return None
    else:
        return pad_mask.ne(1).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)


def get_subsequent_mask(shape, gpu_id):
    """ For masking out the subsequent info. """
    sz_b, len_s = shape
    if gpu_id >= 0:
        subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=gpu_id, dtype=torch.uint8), diagonal=1)
    else:
        subsequent_mask = torch.triu(torch.ones((len_s, len_s), dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask


class Decoder(nn.Module):

    def __init__(self, len_max_seq, n_layers, n_head, d_k, d_v, d_model, d_inner, temporal_dim, spatial_dim, output_dim, dropout=0.1, gpu_id=-1):
        super().__init__()
        n_position = len_max_seq + 1
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_model, padding_idx=0), freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, temporal_dim, spatial_dim, dropout=dropout, gpu_id=gpu_id) for _ in range(n_layers)])
        self.output_fc = nn.Linear(d_model, output_dim)
        self.gpu_id = gpu_id

    def forward(self, dec_inputs, spatial_cache, temporal_cache, temporal_spatial_link, pad_cache, pad_mask=None, return_attns=False, recurrent_steps=1):
        b, t, _ = dec_inputs.shape
        if self.gpu_id >= 0:
            dec_pos = torch.arange(1, t + 1, device=self.gpu_id).repeat(b, 1)
        else:
            dec_pos = torch.arange(1, t + 1).repeat(b, 1)
        dec_outputs = dec_inputs + self.position_enc(dec_pos)
        slf_attn_mask_subseq = get_subsequent_mask((b, t), self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad = get_attn_key_pad_mask(pad_mask, dec_inputs)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = slf_attn_mask_subseq
        dec_enc_attn_mask = get_attn_key_pad_mask(pad_cache, dec_inputs)
        non_pad_mask = get_non_pad_mask(dec_inputs, pad_mask)
        for i in range(recurrent_steps):
            for dec_layer in self.layer_stack:
                dec_outputs, attns = dec_layer(dec_outputs, temporal_cache, spatial_cache, temporal_spatial_link, non_pad_mask, slf_attn_mask=slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)
        dec_outputs = self.output_fc(dec_outputs)
        if return_attns:
            return dec_outputs, attns
        return dec_outputs,


class TemporalCacheEncoder(nn.Module):

    def __init__(self, len_max_seq, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, gpu_id=-1):
        super().__init__()
        n_position = len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_model, padding_idx=0), freeze=True)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.gpu_id = gpu_id

    def forward(self, src_seq, return_attns=False, recurrent_steps=1, pad_mask=None):
        enc_slf_attn_list = []
        b, t, _ = src_seq.shape
        if self.gpu_id >= 0:
            src_pos = torch.arange(1, t + 1, device=self.gpu_id).repeat(b, 1)
        else:
            src_pos = torch.arange(1, t + 1).repeat(b, 1)
        enc_output = src_seq + self.position_enc(src_pos)
        enc_output = self.dropout_emb(enc_output)
        if pad_mask is not None:
            slf_attn_mask = get_attn_key_pad_mask(pad_mask, src_seq)
        else:
            slf_attn_mask = None
        non_pad_mask = get_non_pad_mask(src_seq, pad_mask)
        for i in range(recurrent_steps):
            for enc_layer in self.layer_stack:
                enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask, slf_attn_mask=slf_attn_mask)
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class CNP(nn.Module):

    def __init__(self, tasks, conf=None, domains=['EMPTY'], gpu_id=-1):
        super(CNP, self).__init__()
        default_conf = self.__defaultconf__()
        if conf != None:
            for k in conf.keys():
                if k not in conf:
                    raise ValueError('The provided configuration does not contain %s' % k)
        else:
            conf = default_conf
        self.gpu_id = gpu_id
        self.input_dim = conf['input_dim']
        self.control_dim = conf['control_dim']
        self.output_dim = conf['output_dim']
        self.spatial_dim = conf['spatial_dim']
        self.temporal_dim = conf['temporal_dim']
        self.temporal_n_layers = conf['temporal_n_layers']
        self.temporal_n_heads = conf['temporal_n_heads']
        self.temporal_d_k = conf['temporal_d_k']
        self.temporal_d_v = conf['temporal_d_v']
        self.temporal_hidden_dim = conf['temporal_hidden_dim']
        self.decoder_dim = conf['decoder_dim']
        self.decoder_n_layers = conf['decoder_n_layers']
        self.decoder_n_heads = conf['decoder_n_heads']
        self.decoder_d_k = conf['decoder_d_k']
        self.decoder_d_v = conf['decoder_d_v']
        self.decoder_hidden_dim = conf['decoder_hidden_dim']
        self.max_seq_len = conf['max_seq_len']
        self.output_embedding_dim = conf['output_embedding_dim']
        self.dropout = conf['dropout']
        self.batch_size = -1
        if isinstance(tasks, dict):
            self.task_clflen = list(tasks.values())
            self.task_dict = {t: i for i, t in enumerate(tasks.keys())}
        else:
            raise ValueError('Tasks must be of type dict containing the tasks and output classifier dimension')
        self.output_clfs = nn.ModuleList([nn.Linear(self.output_dim, t) for t in self.task_clflen])
        self.output_embs = nn.ModuleList([nn.Embedding(t + 1, self.output_embedding_dim, padding_idx=t) for t in self.task_clflen])
        control_states = domains + list(tasks.keys())
        self.control_peripheral = ControlPeripheral(self.control_dim, control_states, gpu_id=gpu_id)
        self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len, self.temporal_n_layers, self.temporal_n_heads, self.temporal_d_k, self.temporal_d_v, self.temporal_dim, self.temporal_hidden_dim, dropout=self.dropout, gpu_id=self.gpu_id)
        self.decoder = Decoder(self.max_seq_len, self.decoder_n_layers, self.decoder_n_heads, self.decoder_d_k, self.decoder_d_v, self.decoder_dim, self.decoder_hidden_dim, self.temporal_dim, self.spatial_dim, self.output_dim, dropout=self.dropout, gpu_id=self.gpu_id)
        self.spatial_cache = None
        self.temporal_cache = None
        self.decoder_cache = None
        self.temporal_spatial_link = []
        self.pad_cache = None
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        self.inpcont_input_proj = nn.Linear(self.input_dim + self.control_dim, self.input_dim)
        self.input_spatial_proj = nn.Linear(self.input_dim, self.spatial_dim)
        self.input_temporal_proj = nn.Linear(self.input_dim, self.temporal_dim)
        self.emb_decoder_proj = nn.Linear(self.output_embedding_dim, self.decoder_dim)
        self.cont_decoder_proj = nn.Linear(self.control_dim, self.decoder_dim)

    def decode(self, task, targets=None, num_steps=100, recurrent_steps=1, pad_mask=None, beam_width=1):
        if targets is not None:
            b, t = targets.shape
            if len(targets.shape) != 2 or targets.shape[0] != self.batch_size:
                raise ValueError('Target tensor must be of shape (batch_size,length of sequence).')
            if task not in self.task_dict.keys():
                raise ValueError('Invalid task %s' % task)
            dec_inputs = self.output_embs[self.task_dict[task]](targets)
            dec_inputs = self.emb_decoder_proj(dec_inputs)
            control = self.control_peripheral(task, self.batch_size)
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)
            dec_inputs = torch.cat([control, dec_inputs], 1)
            if pad_mask is not None:
                pad_extra = torch.zeros((b, 1), device=self.gpu_id, dtype=pad_mask.dtype)
                pad_mask = torch.cat([pad_extra, pad_mask], 1)
            logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link, self.pad_cache, recurrent_steps=recurrent_steps, pad_mask=pad_mask)
            predictions = self.output_clfs[self.task_dict[task]](logits)
            predictions = predictions[:, 0:t, :]
            return log_softmax(predictions, dim=2)
        else:
            control = self.control_peripheral(task, self.batch_size)
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)
            dec_inputs = control
            for i in range(num_steps - 1):
                logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link, self.pad_cache, recurrent_steps=recurrent_steps)
                prediction = self.output_clfs[self.task_dict[task]](logits)
                prediction = prediction[:, -1, :].unsqueeze(1)
                prediction = log_softmax(prediction, dim=2).argmax(-1)
                prediction = self.output_embs[self.task_dict[task]](prediction)
                prediction = self.emb_decoder_proj(prediction).detach()
                if beam_width > 1:
                    p = torch.topk(softmax(prediction), beam_width)
                dec_inputs = torch.cat([dec_inputs, prediction], 1)
            logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache, self.temporal_spatial_link, self.pad_cache, recurrent_steps=recurrent_steps)
            predictions = self.output_clfs[self.task_dict[task]](logits)
            return log_softmax(predictions, dim=2)

    def encode(self, input, pad_mask=None, domain='EMPTY', recurrent_steps=1):
        if len(input.shape) != 4:
            raise Exception('Invalid input dimensions.')
        b, t, s, f = list(input.size())
        self.temporal_spatial_link.append((t, s))
        if b != self.batch_size:
            raise Exception('Input batch size does not match.')
        control_vecs = self.control_peripheral(domain, (b, t, s))
        input = torch.cat([input, control_vecs], 3)
        input = self.inpcont_input_proj(input)
        if s > 1:
            spatial_f = torch.reshape(input, [b, t * s, f])
            spatial_f = self.input_spatial_proj(spatial_f)
            if self.spatial_cache is None:
                self.spatial_cache = spatial_f
            else:
                self.spatial_cache = torch.cat([self.spatial_cache, spatial_f], 1)
        temp_data = input.transpose(2, 3).reshape(b * t, f, s)
        temp_data = self.spatial_pool(temp_data).reshape(b, t, f)
        temp_data = self.input_temporal_proj(temp_data)
        temp_data, = self.temporal_encoder(temp_data, pad_mask=pad_mask, recurrent_steps=recurrent_steps)
        if self.temporal_cache is None:
            self.temporal_cache = temp_data
        else:
            self.temporal_cache = torch.cat([self.temporal_cache, temp_data], 1)
        if pad_mask is None:
            pad_mask = torch.zeros((b, t), device=self.gpu_id, dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache = pad_mask
        else:
            self.pad_cache = torch.cat([self.pad_cache, pad_mask], 1)

    def clear_spatial_cache(self):
        self.spatial_cache = None

    def clear_temporal_cache(self):
        self.temporal_raw_cache = None
        self.temporal_cache = None

    def reset(self, batch_size=1):
        self.attn_scores = []
        self.batch_size = batch_size
        self.temporal_spatial_link = []
        self.pad_cache = None
        self.clear_spatial_cache()
        self.clear_temporal_cache()

    @staticmethod
    def __defaultconf__():
        conf = {'input_dim': 128, 'control_dim': 32, 'output_dim': 128, 'spatial_dim': 128, 'temporal_dim': 512, 'temporal_n_layers': 6, 'temporal_n_heads': 8, 'temporal_d_k': 64, 'temporal_d_v': 64, 'temporal_hidden_dim': 2048, 'decoder_dim': 512, 'decoder_n_layers': 6, 'decoder_n_heads': 8, 'decoder_d_k': 64, 'decoder_d_v': 64, 'decoder_hidden_dim': 2048, 'max_seq_len': 1000, 'output_embedding_dim': 300, 'dropout': 0.1}
        return conf


class base_peripheral(nn.Module):
    """
        The base standard non recursive perpheral
        All base peripherals must implement the following functions:
            __init__()
            run_cycle()

    """

    def __init__(self):
        super(base_peripheral, self).__init__()


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ImageInputPeripheral(base_peripheral):

    def __init__(self, output_dim, dropout=0, weights_preload=True, freeze_layers=True):
        self.feature_dim = 2048
        super(ImageInputPeripheral, self).__init__()
        self.image_model = resnet152(pretrained=weights_preload)
        if freeze_layers:
            self.image_model = self.image_model.eval()
            self.image_model.train = self.empty_fun
            self.image_model.eval = self.empty_fun
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.enc_dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(self.feature_dim, output_dim)

    def encode(self, image_tensor):
        shape = image_tensor.shape
        if len(shape) == 5:
            t_dim = image_tensor.shape[1]
            image_tensor = torch.reshape(image_tensor, (-1, 3, shape[3], shape[4]))
        batch_size = image_tensor.shape[0]
        image_enc = self.image_model(image_tensor)
        enc_reshape = torch.reshape(image_enc, [batch_size, self.feature_dim, -1])
        enc_transposed = torch.transpose(enc_reshape, 1, 2)
        drp_enc = self.enc_dropout(enc_transposed)
        output_enc = self.output_fc(drp_enc)
        if len(shape) == 5:
            output_enc = torch.reshape(output_enc, (-1, t_dim, output_enc.shape[1], output_enc.shape[2]))
        else:
            output_enc = output_enc.unsqueeze(1)
        return output_enc

    def empty_fun(self, mode):
        pass


class LanguagePeripheral(base_peripheral):

    def __init__(self, output_dim, vocab_size=10000, embed_dim=50, lang='en', embedding_preload=True, gpu_id=-1, dropout=0):
        super(LanguagePeripheral, self).__init__()
        self.gpu_id = gpu_id
        self.pad_char = vocab_size
        self.bpe_encoder = BPEmb(lang=lang, vs=vocab_size, dim=embed_dim, add_pad_emb=True)
        self.embed_layer = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=self.pad_char)
        if embedding_preload == True:
            self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
            None
        self.enc_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens):
        pad_mask = tokens.eq(self.id_PAD)
        embeddings = self.embed_layer(tokens)
        embeddings = self.enc_dropout(embeddings)
        output = self.output(embeddings)
        return output.unsqueeze(2)

    def embed_sentences(self, sentences):
        tokens, pad_mask = self.tokenize_sentences(sentences)
        return self.forward(tokens), pad_mask

    def decode_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().astype(int).tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.astype(int).tolist()
        filtered_tokens = []
        for t in tokens:
            values = []
            for i in t:
                if i == self.id_EOS:
                    break
                elif i < self.id_PAD:
                    values.append(i)
            filtered_tokens.append(values)
        return self.bpe_encoder.decode_ids(filtered_tokens)

    def tokenize_sentences(self, sentences):
        tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)
        max_len = 0
        for t in tokens:
            max_len = max(max_len, len(t))
        for i in range(len(tokens)):
            tok_len = len(tokens[i])
            tokens[i].extend([self.pad_char] * (max_len - tok_len))
        tokens = torch.tensor(np.array(tokens))
        if self.gpu_id > -1:
            tokens = tokens
        pad_mask = tokens.eq(self.id_PAD)
        return tokens, pad_mask

    @property
    def id_PAD(self):
        return self.pad_char

    @property
    def id_GO(self):
        return 1

    @property
    def id_EOS(self):
        return 2


class OmniNet(nn.Module):

    def __init__(self, config=None, gpu_id=-1, dropout=None):
        super(OmniNet, self).__init__()
        if config is None:
            cc, pc, d = self.__defaultconf__()
        else:
            cc, pc, d = config
        if dropout is not None:
            cc['dropout'] = dropout
            pc['dropout'] = dropout
        self.gpu_id = gpu_id
        tasks = {'PENN': pc['penn_output_classes'], 'HMDB': pc['hmdb_output_classes'], 'IMAGE_CAPTION': pc['english_language_output_vocab'], 'VQA': pc['vqa_output_vocab']}
        self.cnp = CNP(tasks, conf=cc, domains=d, gpu_id=gpu_id)
        self.image_input_perph = ImageInputPeripheral(output_dim=cc['input_dim'], dropout=pc['dropout'], freeze_layers=True)
        self.english_language_perph = LanguagePeripheral(vocab_size=pc['english_language_input_vocab'], embed_dim=pc['english_language_input_embed'], output_dim=cc['input_dim'], lang='en', gpu_id=gpu_id, dropout=pc['dropout'])
        self.german_language_perph = LanguagePeripheral(vocab_size=pc['german_language_input_vocab'], embed_dim=pc['german_language_input_embed'], output_dim=cc['input_dim'], lang='de', gpu_id=gpu_id)

    def reset(self, batch_size):
        self.cnp.reset(batch_size)

    def encode_videos(self, videos, domain='IMAGE'):
        video_encodings = self.image_input_perph.encode(videos)
        self.cnp.encode(video_encodings, domain=domain)

    def encode_images(self, images, domain='IMAGE'):
        image_encodings = self.image_input_perph.encode(images)
        self.cnp.encode(image_encodings, domain=domain)

    def encode_englishtexts(self, texts, domain='ENGLISH'):
        sent_encodings, input_pad_mask = self.english_language_perph.embed_sentences(texts)
        self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain)

    def decode_from_targets(self, task, targets, target_pad_mask=None):
        return self.cnp.decode(task, targets=targets, pad_mask=target_pad_mask)

    def decode_greedy(self, task, num_steps):
        return self.cnp.decode(task, targets=None, num_steps=num_steps)

    def save(self, checkpoint_dir, iterations):
        save_dir = os.path.join(checkpoint_dir, str(iterations))
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pth'))
        None

    def restore(self, checkpoint_dir, iterations):
        save_dir = os.path.join(checkpoint_dir, str(iterations), 'model.pth')
        pretrained_dict = torch.load(save_dir)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
        self.load_state_dict(pretrained_dict, strict=False)
        None

    def restore_file(self, file):
        pretrained_dict = torch.load(file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
        self.load_state_dict(pretrained_dict, strict=False)

    @staticmethod
    def __defaultconf__():
        """
        The default confurigation as specified in the original paper

        """
        cnp_conf = {'input_dim': 512, 'control_dim': 32, 'output_dim': 512, 'spatial_dim': 512, 'temporal_dim': 512, 'temporal_n_layers': 6, 'temporal_n_heads': 8, 'temporal_d_k': 64, 'temporal_d_v': 64, 'temporal_hidden_dim': 2048, 'decoder_dim': 512, 'decoder_n_layers': 6, 'decoder_n_heads': 8, 'decoder_d_k': 64, 'decoder_d_v': 64, 'decoder_hidden_dim': 2048, 'max_seq_len': 500, 'output_embedding_dim': 300, 'dropout': 0.1}
        perph_conf = {'german_language_input_vocab': 25000, 'german_language_input_embed': 300, 'english_language_input_vocab': 25000, 'english_language_input_embed': 300, 'english_language_output_vocab': 25000, 'german_language_output_vocab': 25000, 'dropout': 0.1, 'vqa_output_vocab': 3500, 'hmdb_output_classes': 52, 'penn_output_classes': 48}
        domains = ['ENGLISH', 'GERMAN', 'IMAGE']
        return cnp_conf, perph_conf, domains


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (TemporalCacheEncoder,
     lambda: ([], {'len_max_seq': 4, 'n_layers': 1, 'n_head': 4, 'd_k': 4, 'd_v': 4, 'd_model': 4, 'd_inner': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_subho406_OmniNet(_paritybench_base):
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

