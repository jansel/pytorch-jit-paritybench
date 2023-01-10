import sys
_module = sys.modules[__name__]
del sys
mosi_config = _module
resume_config = _module
run_torch = _module
coco_label = _module
coco_region = _module
torchmm = _module
datasets = _module
basic_dataset = _module
preprocess = _module
bag_of_words = _module
base_tokenizer = _module
build_vocab = _module
n_grams = _module
pre_tokenizer = _module
region_processor = _module
pretrain_dataset = _module
reader = _module
coco_reader = _module
sample_dataset = _module
semi_dataset = _module
engines = _module
base_trainer = _module
caption_trainer = _module
fusion_trainer = _module
mosi_trainer = _module
multitask_trainer = _module
retrieval_trainer = _module
tools = _module
reduce_lr = _module
metrics = _module
caption = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
ciderD = _module
ciderD_scorer = _module
meteor = _module
rouge = _module
spice = _module
tokenizer = _module
ptbtokenizer = _module
fusion = _module
retrieval = _module
score = _module
models = _module
captioning = _module
aoanet = _module
layers = _module
label_smooth = _module
nic = _module
backbone = _module
gru = _module
lstm = _module
resnet = _module
vgg = _module
cmml = _module
early = _module
late = _module
lmf = _module
tmc = _module
multitask = _module
bert_config = _module
vilbert = _module
bfan = _module
domfn = _module
imram = _module
contrastive = _module
img_enc = _module
txt_enc = _module
utils = _module
scan = _module
sgraf = _module
vsepp = _module
start = _module
logger = _module
option = _module

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


from torch.utils.data import Dataset


import torchvision.transforms as T


import random


import numpy as np


import torch.cuda


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import DataLoader


from torch.nn.utils.clip_grad import clip_grad_norm


from scipy import stats


import torch.nn as nn


import torch.optim as optim


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


from sklearn.metrics import roc_auc_score


from sklearn.metrics import mean_absolute_error


from torch.optim.lr_scheduler import ReduceLROnPlateau


import copy


import math


import torch.nn.functional as F


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torchvision.models import resnet152


from torchvision.models import vgg16


from torchvision import models


from torch.nn.parameter import Parameter


from torch.autograd import Variable


from torch.nn.init import xavier_normal


from torch import nn


from torch.nn import CrossEntropyLoss


import torchvision.models as models


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedDotAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        self.d_k = d_model * scale // h
        self.h = h
        self.project_k_v = project_k_v
        self.norm_q = norm_q
        self.norm = LayerNorm(d_model)
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)
        self.use_aoa = do_aoa
        self.do_dropout_aoa = dropout_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            self.dropout_aoa = nn.Dropout(p=dropout_aoa)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)
        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)
        nbatches = query.size(0)
        if self.norm_q:
            query = self.norm(query)
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if self.use_aoa:
            if self.do_dropout_aoa:
                x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
            else:
                x = self.aoa_layer(torch.cat([x, query], -1))
        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class AoA_Refiner_Layer(nn.Module):

    def __init__(self, size, self_attn, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return x


class AoA_Refiner_Core(nn.Module):

    def __init__(self, num_heads, rnn_size):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(num_heads, rnn_size, project_k_v=1, scale=1, do_aoa=1, norm_q=0, dropout_aoa=0.3)
        layer = AoA_Refiner_Layer(rnn_size, attn, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AoA_Decoder_Core(nn.Module):

    def __init__(self, num_heads, rnn_size, drop_prob_lm, input_encoding_size):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = drop_prob_lm
        self.d_model = rnn_size
        self.att_lstm = nn.LSTMCell(input_encoding_size + rnn_size, rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        self.att2ctx = nn.Sequential(nn.Linear(self.d_model + rnn_size, 2 * rnn_size), nn.GLU())
        self.attention = MultiHeadedDotAttention(num_heads, rnn_size, project_k_v=0, scale=1, use_output_layer=0, do_aoa=0, norm_q=1)
        self.ctx_drop = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        att = self.attention(h_att, p_att_feats.narrow(2, 0, self.d_model), p_att_feats.narrow(2, self.d_model, self.d_model), att_masks)
        ctx_input = torch.cat([att, h_att], 1)
        output = self.att2ctx(ctx_input)
        state = torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))
        output = self.out_drop(output)
        return output, state


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)
        self.size = input.size(1)
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AoANet(nn.Module):

    def __init__(self, vocab_size, input_encoding_size, rnn_size, num_layers, drop_prob_lm, sample_len, input_feat_size, num_heads, smoothing, **kwargs):
        super(AoANet, self).__init__()
        self.num_layers = 2
        self.ctx2att = nn.Linear(rnn_size, 2 * rnn_size)
        self.refiner = AoA_Refiner_Core(num_heads, rnn_size)
        self.core = AoA_Decoder_Core(num_heads, rnn_size, drop_prob_lm, input_encoding_size)
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = sample_len
        self.fc_feat_size = input_feat_size
        self.att_feat_size = input_feat_size
        self.use_bn = getattr(kwargs, 'use_bn', 0)
        self.ss_prob = 0.0
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) + (nn.Linear(self.att_feat_size, self.rnn_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)) + ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.criterion = LabelSmoothing(smoothing=smoothing)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.rnn_size), weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def forward_xe(self, fc_feats, att_feats, seq, att_masks, cap_mask):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            if i >= 1 and seq[:, i].sum() == 0:
                break
            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output
        loss = self.criterion(outputs, seq[:, 1:], cap_mask[:, 1:])
        return loss

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)
        if att_masks is None:
            mean_feats = torch.mean(att_feats, dim=1)
        else:
            mean_feats = torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1)
        p_att_feats = self.ctx2att(att_feats)
        return mean_feats, att_feats, p_att_feats, att_masks

    def forward(self, batch, mode='xe', sample_method='greedy'):
        images = batch['image_feat']
        images_loc = batch['image_loc'].float()
        images = torch.cat([images, images_loc], dim=-1)
        images_mask = batch['image_mask']
        captions = batch['text_token']
        cap_mask = batch['text_mask']
        batch_size = images.shape[0]
        seq_len = captions.shape[-1]
        images = images.unsqueeze(1).expand([batch_size, 5, 36, self.att_feat_size]).reshape([batch_size * 5, 36, self.att_feat_size])
        images_mask = images_mask.unsqueeze(1).expand([batch_size, 5, 36]).reshape([batch_size * 5, 36])
        captions = captions.reshape([batch_size * 5, seq_len])
        cap_mask = cap_mask.reshape([batch_size * 5, seq_len])
        if torch.cuda.is_available():
            images = images
            images_mask = images_mask
            captions = captions
            cap_mask = cap_mask
        fc_feats = images.mean(1)
        if self.training:
            if mode == 'xe':
                return self.forward_xe(fc_feats, images, captions, images_mask, cap_mask)
            else:
                return self.forward_sample(fc_feats, images, images_mask, sample_method)
        else:
            return self.forward_sample(fc_feats, images, images_mask, sample_method)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def forward_sample(self, fc_feats, att_feats, att_masks=None, sample_method='greedy'):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            if t == self.seq_length:
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method)
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0:
                break
        return seqLogprobs, seq

    def sample_next_word(self, logprobs, sample_method):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        else:
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        return it, sampleLogprobs


class Encoder(nn.Module):

    def __init__(self, network='vgg16'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = vgg16(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


class Attention(nn.Module):

    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class Decoder(nn.Module):

    def __init__(self, vocab_size, encoder_dim, tf=False):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()
        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.deep_output = nn.Linear(512, vocab_size)
        self.dropout = nn.Dropout()
        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)

    def forward(self, img_features, captions):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html
        """
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)
        max_timespan = captions.size(1) - 1
        prev_words = torch.ones(batch_size, 1).long()
        if torch.cuda.is_available():
            prev_words = prev_words
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)
        preds = []
        alphas = []
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))
            preds.append(output.unsqueeze(1))
            alphas.append(alpha.unsqueeze(1))
            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        preds = torch.cat(preds, 1)
        alphas = torch.cat(alphas, 1)
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)
        c = self.init_c(avg_features)
        c = self.tanh(c)
        h = self.init_h(avg_features)
        h = self.tanh(h)
        return h, c


class NIC(nn.Module):

    def __init__(self, network, vocab_size, teacher_forcing, alpha_c, **kwargs):
        super(NIC, self).__init__()
        self.encoder = Encoder(network=network)
        self.decoder = Decoder(vocab_size=vocab_size, encoder_dim=self.encoder.dim, tf=teacher_forcing)
        self.criterion = nn.CrossEntropyLoss()
        self.alpha_c = alpha_c

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        if torch.cuda.is_available():
            images = images
            captions = captions
        with torch.no_grad():
            images = self.encoder(images)
        logit, alphas = self.decoder(images, captions)
        target = captions[:, 1:captions.size(1)]
        scores = logit.reshape([-1, logit.size(-1)])
        target = target.reshape([-1, 1]).squeeze()
        loss = self.criterion(scores, target)
        loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        if self.training:
            return loss
        else:
            logit = torch.argmax(logit, dim=-1)
            return loss, logit


class Gru(nn.Module):

    def __init__(self, word_dim, hidden_dim, num_labels, vocab_size, model_name):
        super(Gru, self).__init__()
        self.model_name = model_name.lower()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.emb = nn.Embedding(vocab_size, self.input_dim)
        self.hidden = nn.GRU(self.input_dim, self.hidden_dim, 2)
        self.predict = nn.Linear(self.hidden_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        if torch.cuda.is_available():
            batch['text_token'] = batch['text_token']
            batch['label'] = batch['label']
        txt_emb = self.emb(batch['text_token'])
        txt_hidden, y = self.hidden(txt_emb)
        txt_hidden = torch.mean(txt_hidden, 1)
        txt_predict = self.predict(txt_hidden)
        if self.model_name == 'earlyfusion' or self.model_name == 'lmffusion':
            return txt_hidden
        elif self.model_name == 'latefusion' or self.model_name == 'tmcfusion':
            return txt_predict
        else:
            txt_predict = self.sigmoid(txt_predict)
            loss = self.criterion(txt_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, txt_predict


class Lstm(nn.Module):

    def __init__(self, word_dim, hidden_dim, num_labels, vocab_size, model_name):
        super(Lstm, self).__init__()
        self.model_name = model_name.lower()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.emb = nn.Embedding(vocab_size, self.input_dim)
        self.hidden = nn.LSTM(self.input_dim, self.hidden_dim, 2)
        self.predict = nn.Linear(self.hidden_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        if torch.cuda.is_available():
            batch['text_token'] = batch['text_token']
            batch['label'] = batch['label']
        txt_emb = self.emb(batch['text_token'])
        y, (h, c) = self.hidden(txt_emb)
        txt_hidden = torch.mean(y, 1)
        txt_predict = self.predict(txt_hidden)
        if self.model_name == 'earlyfusion' or self.model_name == 'lmffusion':
            return txt_hidden
        elif self.model_name == 'latefusion' or self.model_name == 'tmcfusion':
            return txt_predict
        else:
            txt_predict = self.sigmoid(txt_predict)
            loss = self.criterion(txt_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, txt_predict


class Resnet(nn.Module):

    def __init__(self, hidden_dim, num_labels, model_name, finetune=False):
        super(Resnet, self).__init__()
        self.model_name = model_name.lower()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.hidden = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU())
        self.predict = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        if not finetune:
            for p in self.resnet.parameters():
                p.require_grads = False

    def forward(self, batch):
        if torch.cuda.is_available():
            batch['image_feat'] = batch['image_feat']
            batch['label'] = batch['label']
        img_resnet = self.resnet(batch['image_feat'])
        img_resnet = torch.squeeze(img_resnet, 2)
        img_resnet = torch.squeeze(img_resnet, 2)
        img_hidden = self.hidden(img_resnet)
        img_predict = self.predict(img_hidden)
        if self.model_name == 'earlyfusion' or self.model_name == 'lmffusion':
            return img_hidden
        elif self.model_name == 'latefusion' or self.model_name == 'tmcfusion':
            return img_predict
        else:
            img_predict = self.sigmoid(img_predict)
            loss = self.criterion(img_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, img_predict


class Vgg(nn.Module):

    def __init__(self, hidden_dim, num_labels, model_name, finetune=False):
        super(Vgg, self).__init__()
        self.model_name = model_name.lower()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.hidden = nn.Sequential(nn.Linear(4096, hidden_dim), nn.ReLU())
        self.predict = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        if not finetune:
            for p in self.vgg.parameters():
                p.require_grads = False

    def forward(self, batch):
        if torch.cuda.is_available():
            batch['image_feat'] = batch['image_feat']
            batch['label'] = batch['label']
        img_vgg = self.vgg(batch['image_feat'])
        img_hidden = self.hidden(img_vgg)
        img_predict = self.predict(img_hidden)
        if self.model_name == 'earlyfusion' or self.model_name == 'lmffusion':
            return img_hidden
        elif self.model_name == 'latefusion' or self.model_name == 'tmcfusion':
            return img_predict
        else:
            img_predict = self.sigmoid(img_predict)
            loss = self.criterion(img_predict, batch['label'])
            if self.training:
                return loss
            else:
                return loss, img_predict


class CMML(nn.Module):

    def __init__(self, bow_dim, hidden_dim, num_labels, **kwargs):
        super(CMML, self).__init__()
        input_dim = bow_dim
        hidden_dim = hidden_dim
        num_labels = num_labels
        self.txt_hidden = nn.Linear(input_dim, hidden_dim)
        self.txt_hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.txt_predict = nn.Linear(hidden_dim, num_labels)
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.img_hidden = nn.Linear(512, hidden_dim)
        self.img_predict = nn.Linear(hidden_dim, num_labels)
        self.attn_mlp = nn.Linear(hidden_dim, 1)
        self.attn_mlp = nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim / 2)), nn.ReLU(), nn.Linear(int(hidden_dim / 2), 1))
        self.modality_predict = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.BCELoss()
        self.cita = 1.003

    def pretrain(self, batch):
        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img, unsupervised_txt, unsupervised_img = feature
        supervised_txt = torch.cat(supervised_txt, 0)
        supervised_img = torch.cat(supervised_img, 0)
        label = torch.cat(label, 0)
        if torch.cuda.is_available():
            supervised_txt = supervised_txt
            supervised_img = supervised_img
            label = label
        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        supervised_img_predict = self.img_predict(supervised_img_hidden)
        supervised_img_predict = self.sigmoid(supervised_img_predict)
        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_txt_predict = self.txt_predict(supervised_txt_hidden)
        supervised_txt_predict = self.sigmoid(supervised_txt_predict)
        img_loss = self.criterion(supervised_img_predict, label)
        txt_loss = self.criterion(supervised_txt_predict, label)
        loss = img_loss + txt_loss
        return loss

    def forward(self, batch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_train(self, batch):
        """for training, batch contain supervised and unsupervised data"""
        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img, unsupervised_txt, unsupervised_img = feature
        supervised_txt = torch.cat(supervised_txt, 0)
        supervised_img = torch.cat(supervised_img, 0)
        unsupervised_txt = torch.cat(unsupervised_txt, 0)
        unsupervised_img = torch.cat(unsupervised_img, 0)
        label = torch.cat(label, 0)
        if torch.cuda.is_available():
            supervised_txt = supervised_txt
            supervised_img = supervised_img
            unsupervised_txt = unsupervised_txt
            unsupervised_img = unsupervised_img
            label = label
        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_txt_predict = self.txt_predict(supervised_txt_hidden)
        supervised_txt_predict = self.sigmoid(supervised_txt_predict)
        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        supervised_img_predict = self.img_predict(supervised_img_hidden)
        supervised_img_predict = self.sigmoid(supervised_img_predict)
        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = torch.cat([attn_txt, attn_img], dim=1)
        attn_modality = self.softmax(attn_modality)
        attn_img = torch.zeros([1, len(label)])
        attn_txt = torch.zeros([1, len(label)])
        if torch.cuda.is_available():
            attn_txt = attn_txt
            attn_img = attn_img
        attn_img[0] = attn_modality[:, 0]
        attn_img = torch.t(attn_img)
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = torch.t(attn_txt)
        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)
        mm_loss = self.criterion(supervised_predict, label)
        txt_loss = self.criterion(supervised_txt_predict, label)
        img_loss = self.criterion(supervised_img_predict, label)
        supervised_loss = mm_loss * 3 + txt_loss + img_loss
        similar = torch.bmm(supervised_img_predict.unsqueeze(1), supervised_txt_predict.unsqueeze(2))
        similar = torch.reshape(similar, shape=[supervised_img_predict.shape[0]])
        norm_matrix_img = torch.norm(supervised_img_predict, p=2, dim=1)
        norm_matrix_text = torch.norm(supervised_txt_predict, p=2, dim=1)
        div = torch.mean(similar / (norm_matrix_img * norm_matrix_text))
        unsupervised_txt_hidden = self.txt_hidden(unsupervised_txt)
        unsupervised_txt_predict = self.txt_predict(unsupervised_txt_hidden)
        unsupervised_txt_predict = self.sigmoid(unsupervised_txt_predict)
        unsupervised_img_hidden = self.resnet(unsupervised_img)
        unsupervised_img_hidden = torch.reshape(unsupervised_img_hidden, shape=[unsupervised_img_hidden.shape[0], 512])
        unsupervised_img_hidden = self.img_hidden(unsupervised_img_hidden)
        unsupervised_img_predict = self.img_predict(unsupervised_img_hidden)
        unsupervised_img_predict = self.sigmoid(unsupervised_img_predict)
        unsimilar = torch.bmm(unsupervised_img_predict.unsqueeze(1), unsupervised_txt_predict.unsqueeze(2))
        unsimilar = torch.reshape(unsimilar, shape=[unsupervised_img_predict.shape[0]])
        unnorm_matrix_img = torch.norm(unsupervised_img_predict, p=2, dim=1)
        unnorm_matrix_text = torch.norm(unsupervised_txt_predict, p=2, dim=1)
        dis = 2 - unsimilar / (unnorm_matrix_img * unnorm_matrix_text)
        tensor1 = dis[torch.abs(dis) < self.cita]
        tensor2 = dis[torch.abs(dis) >= self.cita]
        tensor1loss = torch.sum(tensor1 * tensor1 / 2)
        tensor2loss = torch.sum(self.cita * (torch.abs(tensor2) - 1 / 2 * self.cita))
        unsupervised_loss = (tensor1loss + tensor2loss) / unsupervised_img.shape[0]
        total_loss = supervised_loss + 0.01 * div + unsupervised_loss
        return total_loss

    def forward_eval(self, batch):
        """for evaluation, batch only contain supervised data"""
        feature, label = batch['feature'], batch['label']
        supervised_txt, supervised_img = feature
        if torch.cuda.is_available():
            supervised_txt = supervised_txt
            supervised_img = supervised_img
            label = label
        supervised_txt_hidden = self.txt_hidden(supervised_txt)
        supervised_img_hidden = self.resnet(supervised_img)
        supervised_img_hidden = torch.reshape(supervised_img_hidden, shape=[supervised_img_hidden.shape[0], 512])
        supervised_img_hidden = self.img_hidden(supervised_img_hidden)
        attn_txt = self.attn_mlp(supervised_txt_hidden)
        attn_img = self.attn_mlp(supervised_img_hidden)
        attn_modality = torch.cat([attn_txt, attn_img], dim=1)
        attn_modality = self.softmax(attn_modality)
        attn_img = torch.zeros([1, len(label)])
        attn_txt = torch.zeros([1, len(label)])
        if torch.cuda.is_available():
            attn_txt = attn_txt
            attn_img = attn_img
        attn_img[0] = attn_modality[:, 0]
        attn_img = torch.t(attn_img)
        attn_txt[0] = attn_modality[:, 1]
        attn_txt = torch.t(attn_txt)
        supervised_hidden = attn_txt * supervised_txt_hidden + attn_img * supervised_img_hidden
        supervised_predict = self.modality_predict(supervised_hidden)
        supervised_predict = self.sigmoid(supervised_predict)
        total_loss = self.criterion(supervised_predict, label)
        return total_loss, supervised_predict


ModelMap = {'vgg': Vgg, 'resnet': Resnet, 'lstm': Lstm, 'gru': Gru}


class EarlyFusion(nn.Module):

    def __init__(self, image_model, text_model, word_dim, vocab_size, hidden_dim, num_labels, model_name, option, finetune, **kwargs):
        super(EarlyFusion, self).__init__()
        self.option = option
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)
        self.linear1 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_labels))
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        img_hidden = self.image_model(batch)
        txt_hidden = self.text_model(batch)
        if self.option == 'concat':
            hidden = torch.cat([img_hidden, txt_hidden], 1)
            predict = self.linear1(hidden)
        else:
            hidden = torch.add(img_hidden, txt_hidden)
            predict = self.linear2(hidden)
        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])
        if self.training:
            return loss
        else:
            return loss, predict


class LateFusion(nn.Module):

    def __init__(self, image_model, text_model, word_dim, vocab_size, hidden_dim, num_labels, model_name, option, finetune, **kwargs):
        super(LateFusion, self).__init__()
        self.option = option
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        img_predict = self.image_model(batch)
        txt_predict = self.text_model(batch)
        if self.option == 'mean':
            predict = torch.cat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], 0)
            predict = torch.mean(predict, 0)
        else:
            temp = torch.cat([img_predict.unsqueeze(0), txt_predict.unsqueeze(0)], 0)
            sel = torch.abs(temp - 0.5)
            sel_idx = torch.argmax(sel, dim=0)
            predict = (1 - sel_idx) * img_predict + sel_idx * txt_predict
        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])
        if self.training:
            return loss
        else:
            return loss, predict


class LMFFusion(nn.Module):

    def __init__(self, image_model, text_model, word_dim, vocab_size, hidden_dim, num_labels, model_name, finetune, **kwargs):
        super(LMFFusion, self).__init__()
        self.rank = kwargs['rank']
        self.num_labels = num_labels
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)
        self.image_factor = Parameter(torch.Tensor(self.rank, hidden_dim + 1, num_labels))
        self.text_factor = Parameter(torch.Tensor(self.rank, hidden_dim + 1, num_labels))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, num_labels))
        xavier_normal(self.image_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        DTYPE = torch.FloatTensor
        img_predict = self.image_model(batch)
        txt_predict = self.text_model(batch)
        batch_size = txt_predict.data.shape[0]
        img_predict = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), img_predict), dim=1)
        txt_predict = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), txt_predict), dim=1)
        fusion_img = torch.matmul(img_predict, self.image_factor)
        fusion_txt = torch.matmul(txt_predict, self.text_factor)
        fusion_zy = fusion_img * fusion_txt
        predict = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        predict = predict.view(-1, self.num_labels)
        predict = self.sigmoid(predict)
        loss = self.criterion(predict, batch['label'])
        if self.training:
            return loss
        else:
            return loss, predict


def KL(alpha, c):
    beta = torch.ones((1, c))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    A = torch.sum(p * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp, c)
    return A + B


class TMCFusion(nn.Module):

    def __init__(self, image_model, text_model, word_dim, vocab_size, hidden_dim, num_labels, lambda_epochs, model_name, finetune, **kwargs):
        super(TMCFusion, self).__init__()
        self.num_labels = num_labels
        self.lambda_epochs = lambda_epochs
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)
        self.sigmoid = nn.Sigmoid()

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / S[v].expand(E[v].shape)
                u[v] = self.num_labels / S[v]
            bb = torch.bmm(b[0].view(-1, self.num_labels, 1), b[1].view(-1, 1, self.num_labels))
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / (1 - C).view(-1, 1).expand(b[0].shape)
            u_a = torch.mul(u[0], u[1]) / (1 - C).view(-1, 1).expand(u[0].shape)
            S_a = self.num_labels / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a
        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, batch):
        img_hidden = self.image_model(batch)
        txt_hidden = self.text_model(batch)
        evidence = dict()
        evidence[0] = img_hidden
        evidence[1] = txt_hidden
        loss = 0
        alpha = dict()
        for i in range(2):
            alpha[i] = evidence[i] + 1
            loss += ce_loss(batch['label'], alpha[i], self.num_labels, batch['epoch'], self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(batch['label'], alpha_a, self.num_labels, batch['epoch'], self.lambda_epochs)
        loss = torch.mean(loss)
        predict = self.sigmoid(evidence_a)
        if self.training:
            return loss
        else:
            return loss, predict


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.task_specific_tokens = config.task_specific_tokens
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.task_specific_tokens:
            self.task_embeddings = nn.Embedding(20, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, task_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.task_specific_tokens:
            task_embeddings = self.task_embeddings(task_ids)
            embeddings = torch.cat([embeddings[:, 0:1], task_embeddings, embeddings[:, 1:]], dim=1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.v_hidden_size, config.v_num_attention_heads))
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(config.v_hidden_size / config.v_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)
        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))
            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertImageSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):

    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask, txt_embedding, txt_attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):

    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):

    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):

    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask, txt_embedding, txt_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):

    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.bi_hidden_size, config.bi_num_attention_heads))
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)
        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None, use_co_attention_mask=False):
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class BertBiOutput(nn.Module):

    def __init__(self, config):
        super(BertBiOutput, self).__init__()
        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)
        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)
        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)
        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)
        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):

    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)
        self.biOutput = BertBiOutput(config)
        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)
        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None, use_co_attention_mask=False):
        bi_output1, bi_output2, co_attention_probs = self.biattention(input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask, use_co_attention_mask)
        attention_output1, attention_output2 = self.biOutput(bi_output2, input_tensor1, bi_output1, input_tensor2)
        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)
        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)
        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.v_layer = nn.ModuleList([copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)])
        self.c_layer = nn.ModuleList([copy.deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))])

    def forward(self, txt_embedding, image_embedding, txt_attention_mask, txt_attention_mask2, image_attention_mask, co_attention_mask=None, output_all_encoded_layers=True, output_all_attention_masks=False):
        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []
        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []
        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()
        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):
            v_end = v_layer_id
            t_end = t_layer_id
            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end
            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)
            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)
            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2)
                    v_start = self.fixed_v_layer
                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)
            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2)
                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)
            if count == 0 and self.in_batch_pairs:
                image_embedding = image_embedding.unsqueeze(0).expand(batch_size, batch_size, num_regions, v_hidden_size).contiguous().view(batch_size * batch_size, num_regions, v_hidden_size)
                image_attention_mask = image_attention_mask.unsqueeze(0).expand(batch_size, batch_size, 1, 1, num_regions).contiguous().view(batch_size * batch_size, 1, 1, num_regions)
                txt_embedding = txt_embedding.unsqueeze(1).expand(batch_size, batch_size, num_words, t_hidden_size).contiguous().view(batch_size * batch_size, num_words, t_hidden_size)
                txt_attention_mask = txt_attention_mask.unsqueeze(1).expand(batch_size, batch_size, 1, 1, num_words).contiguous().view(batch_size * batch_size, 1, 1, num_words)
                co_attention_mask = co_attention_mask.unsqueeze(1).expand(batch_size, batch_size, 1, num_regions, num_words).contiguous().view(batch_size * batch_size, 1, num_regions, num_words)
            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(image_embedding.size(0), txt_embedding.size(1), txt_embedding.size(2))
                txt_attention_mask = txt_attention_mask.expand(image_embedding.size(0), txt_attention_mask.size(1), txt_attention_mask.size(2), txt_attention_mask.size(3))
            if self.with_coattention:
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[count](image_embedding, image_attention_mask, txt_embedding, txt_attention_mask, co_attention_mask, use_co_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)
            v_start = v_end
            t_start = t_end
            count += 1
            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)
        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2)
            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)
        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)
            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)
        return all_encoder_layers_t, all_encoder_layers_v, (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c)


class BertTextPooler(nn.Module):

    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):

    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImgPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertImagePredictionHead(nn.Module):

    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v):
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False
        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)
        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.task_specific_tokens = config.task_specific_tokens
        self.v_embeddings = BertImageEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)
        self.apply(self.init_weights)

    def forward(self, input_txt, input_imgs, image_loc, token_type_ids=None, attention_mask=None, image_attention_mask=None, co_attention_mask=None, task_ids=None, output_all_encoded_layers=False, output_all_attention_masks=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(input_imgs.size(0), input_imgs.size(1)).type_as(input_txt)
        if self.task_specific_tokens:
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask2 = attention_mask.unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask2 = extended_attention_mask2
        extended_image_attention_mask = extended_image_attention_mask
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
        if co_attention_mask is None:
            co_attention_mask = torch.zeros(input_txt.size(0), input_imgs.size(1), input_txt.size(1)).type_as(extended_image_attention_mask)
        extended_co_attention_mask = co_attention_mask.unsqueeze(1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask
        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(embedding_output, v_embedding_output, extended_attention_mask, extended_attention_mask2, extended_image_attention_mask, extended_co_attention_mask, output_all_encoded_layers=output_all_encoded_layers, output_all_attention_masks=output_all_attention_masks)
        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]
        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)
        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]
        return encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask


class VILBERTPretrain(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self, config):
        super(VILBERTPretrain, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)
        self.visual_target = config.visual_target
        self.num_negative = config.num_negative
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        if self.visual_target == 0:
            self.vis_criterion = nn.KLDivLoss(reduction='none')
        elif self.visual_target == 1:
            self.vis_criterion = nn.MSELoss(reduction='none')
        elif self.visual_target == 2:
            self.vis_criterion = CrossEntropyLoss()

    def forward(self, input_ids, image_feat, image_loc, token_type_ids=None, attention_mask=None, image_attention_mask=None, masked_lm_labels=None, image_label=None, image_target=None, next_sentence_label=None, output_all_attention_masks=False):
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(input_ids, image_feat, image_loc, token_type_ids, attention_mask, image_attention_mask, output_all_encoded_layers=False, output_all_attention_masks=output_all_attention_masks)
        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v)
        if masked_lm_labels is not None and next_sentence_label is not None and image_target is not None:
            prediction_scores_v = prediction_scores_v[:, 1:]
            if self.visual_target == 1:
                img_loss = self.vis_criterion(prediction_scores_v, image_target)
                masked_img_loss = torch.sum(img_loss * (image_label == 1).unsqueeze(2).float()) / max(torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1)
            elif self.visual_target == 0:
                img_loss = self.vis_criterion(F.log_softmax(prediction_scores_v, dim=2), image_target)
                masked_img_loss = torch.sum(img_loss * (image_label == 1).unsqueeze(2).float()) / max(torch.sum(image_label == 1), 0)
            elif self.visual_target == 2:
                num_negative = self.num_negative
                num_across_batch = int(self.num_negative * 0.7)
                num_inside_batch = int(self.num_negative * 0.3)
                batch_size, num_regions, _ = prediction_scores_v.size()
                assert batch_size != 0
                row_across_index = input_ids.new(batch_size, num_regions, num_across_batch).random_(0, batch_size - 1)
                col_across_index = input_ids.new(batch_size, num_regions, num_across_batch).random_(0, num_regions)
                for i in range(batch_size - 1):
                    row_across_index[i][row_across_index[i] == i] = batch_size - 1
                final_across_index = row_across_index * num_regions + col_across_index
                row_inside_index = input_ids.new(batch_size, num_regions, num_inside_batch).zero_()
                col_inside_index = input_ids.new(batch_size, num_regions, num_inside_batch).random_(0, num_regions - 1)
                for i in range(batch_size):
                    row_inside_index[i] = i
                for i in range(num_regions - 1):
                    col_inside_index[:, i, :][col_inside_index[:, i, :] == i] = num_regions - 1
                final_inside_index = row_inside_index * num_regions + col_inside_index
                final_index = torch.cat((final_across_index, final_inside_index), dim=2)
                predict_v = prediction_scores_v[image_label == 1]
                neg_index_v = final_index[image_label == 1]
                flat_image_target = image_target.view(batch_size * num_regions, -1)
                negative_v = flat_image_target[neg_index_v]
                positive_v = image_target[image_label == 1]
                sample_v = torch.cat((positive_v.unsqueeze(1), negative_v), dim=1)
                score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
                masked_img_loss = self.vis_criterion(score, input_ids.new(score.size(0)).zero_())
            masked_lm_loss = self.loss_fct(prediction_scores_t.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = self.loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return masked_lm_loss.unsqueeze(0), masked_img_loss.unsqueeze(0), next_sentence_loss.unsqueeze(0)
        else:
            return prediction_scores_t, prediction_scores_v, seq_relationship_score


class SimpleClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(nn.Linear(in_dim, hid_dim), GeLU(), nn.LayerNorm(hid_dim, eps=1e-12), nn.Linear(hid_dim, out_dim))

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class VILBERTFinetune(BertPreTrainedModel):

    def __init__(self, config, num_labels, dropout_prob=0.1, default_gpu=True):
        super(VILBERTFinetune, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.vil_prediction = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, 3129, 0.5)
        self.vil_prediction_gqa = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, 1533, 0.5)
        self.vil_binary_prediction = SimpleClassifier(config.bi_hidden_size * 2, config.bi_hidden_size * 2, 2, 0.5)
        self.vil_logit = nn.Linear(config.bi_hidden_size, 1)
        self.vil_tri_prediction = nn.Linear(config.bi_hidden_size, 3)
        self.vision_logit = nn.Linear(config.v_hidden_size, 1)
        self.linguisic_logit = nn.Linear(config.hidden_size, 1)
        self.fusion_method = config.fusion_method
        self.apply(self.init_weights)

    def forward(self, input_txt, input_imgs, image_loc, token_type_ids=None, attention_mask=None, image_attention_mask=None, co_attention_mask=None, task_ids=None, output_all_encoded_layers=False, output_all_attention_masks=False):
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(input_txt, input_imgs, image_loc, token_type_ids, attention_mask, image_attention_mask, co_attention_mask, task_ids, output_all_encoded_layers=output_all_encoded_layers, output_all_attention_masks=output_all_attention_masks)
        linguisic_prediction, vision_prediction, vil_binary_prediction = self.cls(sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v)
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False
        vil_prediction = self.vil_prediction(pooled_output)
        vil_prediction_gqa = self.vil_prediction_gqa(pooled_output)
        if pooled_output.size(0) % 2 == 0:
            vil_binary_prediction = self.vil_binary_prediction(pooled_output.view(-1, pooled_output.size(1) * 2))
        vil_logit = self.vil_logit(pooled_output)
        vil_tri_prediction = self.vil_tri_prediction(pooled_output)
        vision_logit = self.vision_logit(self.dropout(sequence_output_v)) + ((1.0 - image_attention_mask) * -10000.0).unsqueeze(2)
        linguisic_logit = self.linguisic_logit(self.dropout(sequence_output_t))
        return vil_prediction, vil_logit, vil_binary_prediction


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clip(min=0)
        cost_im = (self.margin + scores - d2).clip(min=0)
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


def l2norm(X, dim, eps=1e-08):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImageFull(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True, cnn_type='vgg19', finetune=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        if cnn_type.startswith('vgg'):
            self.cnn = models.vgg19(pretrained=True)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
            for param in self.cnn.parameters():
                param.requires_grad = finetune
            self.fc = nn.Linear(4096, 1024)
        elif cnn_type.startswith('resnet'):
            self.cnn = models.resnet152(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            for param in self.cnn.parameters():
                param.requires_grad = finetune
            self.fc = nn.Linear(2048, 1024)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images).squeeze()
        features = l2norm(features, 1)
        features = self.fc(features)
        if self.image_norm:
            features = l2norm(features, 1)
        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, image_norm=True, cnn_type=None, finetune=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.image_norm = image_norm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.image_norm:
            features = l2norm(features, dim=-1)
        return features


def EncoderImage(model_name, img_dim, embed_size, image_norm=True, cnn_type=None, finetune=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    model_name = model_name.lower()
    EncoderMap = {'scan': EncoderImagePrecomp, 'vsepp': EncoderImageFull, 'sgraf': EncoderImagePrecomp, 'imram': EncoderImagePrecomp, 'bfan': EncoderImagePrecomp}
    if model_name in EncoderMap:
        img_enc = EncoderMap[model_name](img_dim, embed_size, image_norm, cnn_type, finetune)
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    return img_enc


class EncoderTextGlobal(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
        super(EncoderTextGlobal, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1)
        out = torch.gather(padded[0], 1, I).squeeze(1)
        out = l2norm(out, 1)
        if self.text_norm:
            out = l2norm(out, dim=-1)
        return out


class EncoderTextRegion(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
        super(EncoderTextRegion, self).__init__()
        self.embed_size = embed_size
        self.text_norm = text_norm
        self.use_bi_gru = use_bi_gru
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        if dropout:
            self.use_drop = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.use_drop = False
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        x = self.embed(x)
        if self.use_drop:
            x = self.dropout(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2
        if self.text_norm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb


def EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, text_norm=True, dropout=0.0):
    """A wrapper to text encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    model_name = model_name.lower()
    EncoderMap = {'scan': EncoderTextRegion, 'vsepp': EncoderTextGlobal, 'sgraf': EncoderTextRegion, 'imram': EncoderTextRegion, 'bfan': EncoderTextRegion, 'pfan': EncoderTextRegion}
    if model_name in EncoderMap:
        txt_enc = EncoderMap[model_name](vocab_size, word_dim, embed_size, num_layers, use_bi_gru, text_norm, dropout)
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    return txt_enc


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def l1norm(X, dim, eps=1e-08):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, raw_feature_norm, smooth, eps=1e-08):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == 'softmax':
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == 'l2norm':
        attn = l2norm(attn, 2)
    elif raw_feature_norm == 'clipped_l2norm':
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == 'l1norm':
        attn = l1norm(attn, 2)
    elif raw_feature_norm == 'clipped_l1norm':
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == 'clipped':
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == 'no_norm':
        pass
    else:
        raise ValueError('unknown first norm type:', raw_feature_norm)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    return weightedContext, attnT


def xattn_score(images, captions, cap_lens, focal_type, lambda_softmax):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        weiContext = func_attention(cap_i_expand, images, focal_type, lambda_softmax)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
        weiContext = func_attention(images, cap_i_expand, focal_type, lambda_softmax)
        i2t_sim = cosine_similarity(images, weiContext, dim=2)
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
        sim = t2i_sim + i2t_sim
        similarities.append(sim)
    similarities = torch.cat(similarities, 1)
    return similarities


class BFAN(nn.Module):
    """
    Bidirectional Focal Attention Network (BFAN) model
    """

    def __init__(self, model_name, embed_size, vocab_size, word_dim, num_layers, image_dim, margin, max_violation, focal_type, lambda_softmax, use_bi_gru=True, image_norm=True, text_norm=True, **kwargs):
        super(BFAN, self).__init__()
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.focal_type = focal_type
        self.lambda_softmax = lambda_softmax

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = cap_lens.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = cap_lens.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        score = xattn_score(img_emb, cap_emb, cap_lens, self.focal_type, self.lambda_softmax)
        loss = self.criterion(score)
        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        shard_size = kwargs.get('shard_size', 128)
        focal_type = model.focal_type
        lambda_softmax = model.lambda_softmax
        n_im_shard = int((len(img_embs) - 1) / shard_size + 1)
        n_cap_shard = int((len(cap_embs) - 1) / shard_size + 1)
        sims = np.zeros((len(img_embs), len(cap_embs)))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))
                with torch.no_grad():
                    im = torch.FloatTensor(img_embs[im_start:im_end])
                    ca = torch.FloatTensor(cap_embs[cap_start:cap_end])
                    l = cap_lens[cap_start:cap_end].tolist()
                    if torch.cuda.is_available():
                        im = im
                        ca = ca
                sim = xattn_score(im, ca, l, focal_type, lambda_softmax)
                sims[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
        return sims


class Fcn(nn.Module):

    def __init__(self, input_dim, feature_dim, num_labels):
        """
        :param input_dim:
        :param feature_dim:
        :param num_labels:
        """
        super(Fcn, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.feature_layer = nn.Sequential(nn.Linear(self.input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.feature_dim), nn.ReLU())
        self.prediction_layer = nn.Sequential(nn.Linear(self.feature_dim, self.num_labels), nn.Sigmoid())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, labels):
        feature = self.feature_layer(batch)
        logit = self.prediction_layer(feature)
        loss = self.criterion(logit, labels)
        return feature, logit, loss


class DOMFN(nn.Module):

    def __init__(self, text_hidden_dim=None, vision_hidden_dim=None, audio_hidden_dim=None, feature_dim=None, num_labels=None, fusion='concat'):
        """
        :param text_hidden_dim:
        :param vision_hidden_dim:
        :param audio_hidden_dim:
        :param feature_dim:
        :param num_labels:
        :param fusion:
        """
        super(DOMFN, self).__init__()
        self.text_hidden_dim = text_hidden_dim
        self.vision_hidden_dim = vision_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.fusion = fusion
        self.text_encoder = Fcn(self.text_hidden_dim, self.feature_dim, self.num_labels)
        self.vision_encoder = Fcn(self.vision_hidden_dim, self.feature_dim, self.num_labels)
        self.audio_encoder = Fcn(self.audio_hidden_dim, self.feature_dim, self.num_labels)
        self.multi_encoder = nn.Sequential(nn.Linear((3 if self.fusion == 'concat' else 1) * self.feature_dim, 64), nn.ReLU(), nn.Linear(64, self.num_labels), nn.Sigmoid())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_embeds=None, vision_embeds=None, audio_embeds=None, labels=None, eta=None):
        text_feat, text_logit, text_celoss = self.text_encoder(text_embeds, labels)
        vision_feat, vision_logit, vision_celoss = self.vision_encoder(vision_embeds, labels)
        audio_feat, audio_logit, audio_celoss = self.audio_encoder(audio_embeds, labels)
        multi_logit = None
        if self.fusion == 'concat':
            multi_feat = torch.cat([text_feat, vision_feat, audio_feat], dim=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'mean':
            multi_feat = torch.mean(torch.stack([text_feat, vision_feat, audio_feat], dim=1), dim=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'max':
            multi_feat, _ = torch.max(torch.stack([text_feat, vision_feat, audio_feat], dim=1), dim=1)
            multi_logit = self.multi_encoder(multi_feat)
        multi_celoss = self.criterion(multi_logit, labels)
        average_logit = torch.mean(torch.stack([text_logit, vision_logit, audio_logit], dim=1), dim=1)
        text_penalty, _ = torch.max((text_logit - average_logit) ** 2, dim=1)
        vision_penalty, _ = torch.max((vision_logit - average_logit) ** 2, dim=1)
        audio_penalty, _ = torch.max((audio_logit - average_logit) ** 2, dim=1)
        text_loss = 0.5 * text_celoss * text_celoss - eta * text_penalty
        vision_loss = 0.5 * vision_celoss * vision_celoss - eta * vision_penalty
        audio_loss = 0.5 * audio_celoss * audio_celoss - eta * audio_penalty
        multi_loss = text_celoss + vision_celoss + audio_celoss + multi_celoss
        return {'text_logit': text_logit, 'vision_logit': vision_logit, 'audio_logit': audio_logit, 'multi_logit': multi_logit, 'text_celoss': text_celoss, 'vision_celoss': vision_celoss, 'audio_celoss': audio_celoss, 'text_penalty': text_penalty, 'vision_penalty': vision_penalty, 'audio_penalty': audio_penalty, 'text_loss': text_loss, 'vision_loss': vision_loss, 'audio_loss': audio_loss, 'multi_loss': multi_loss}


class IMRAM(nn.Module):

    def __init__(self, model_name, embed_size, vocab_size, word_dim, num_layers, image_dim, margin, max_violation, model_mode, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, iteration_step, use_bi_gru=True, image_norm=True, text_norm=True, **kwargs):
        super(IMRAM, self).__init__()
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.linear_t2i = nn.Linear(embed_size * 2, embed_size)
        self.gate_t2i = nn.Linear(embed_size * 2, embed_size)
        self.linear_i2t = nn.Linear(embed_size * 2, embed_size)
        self.gate_i2t = nn.Linear(embed_size * 2, embed_size)
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.model_mode = model_mode
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax
        self.agg_func = agg_func
        self.iteration_step = iteration_step
        self.raw_feature_norm = raw_feature_norm

    def gated_memory_t2i(self, input_0, input_1):
        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_t2i(input_cat))
        gate = torch.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)
        return output

    def gated_memory_i2t(self, input_0, input_1):
        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_i2t(input_cat))
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)
        return output

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        lengths = lengths.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, lengths

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        cap_lens = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = cap_lens.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        if self.model_mode == 'full_IMRAM':
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_t2i + scores_i2t
        elif self.model_mode == 'image_IMRAM':
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_i2t
        elif self.model_mode == 'text_IMRAM':
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            score = scores_t2i
        else:
            raise ValueError('No such mode!')
        loss = self.criterion(score)
        return loss

    def xattn_score_Text_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            query = cap_i_expand
            context = images
            for j in range(self.iteration_step):
                attn_feat, _ = func_attention(query, context, self.raw_feature_norm, smooth=self.lambda_softmax)
                row_sim = cosine_similarity(cap_i_expand, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)
                query = self.gated_memory_t2i(query, attn_feat)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1)
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)
        return new_similarities

    def xattn_score_Image_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            query = images
            context = cap_i_expand
            for j in range(self.iteration_step):
                attn_feat, _ = func_attention(query, context, self.raw_feature_norm, smooth=self.lambda_softmax)
                row_sim = cosine_similarity(images, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)
                query = self.gated_memory_i2t(query, attn_feat)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1)
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)
        return new_similarities

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):

        def shard_xattn_Full_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1
            d_t2i = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
            d_i2t = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = torch.from_numpy(images[im_start:im_end])
                    s = torch.from_numpy(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end]
                    sim_list_t2i = model.xattn_score_Text_IMRAM(im_emb, s, l)
                    sim_list_i2t = model.xattn_score_Image_IMRAM(im_emb, s, l)
                    assert len(sim_list_t2i) == iteration_step and len(sim_list_i2t) == iteration_step
                    for k in range(iteration_step):
                        d_t2i[k][im_start:im_end, cap_start:cap_end] = sim_list_t2i[k].data.cpu().numpy()
                        d_i2t[k][im_start:im_end, cap_start:cap_end] = sim_list_i2t[k].data.cpu().numpy()
            score = 0
            for j in range(iteration_step):
                score += d_t2i[j]
            for j in range(iteration_step):
                score += d_i2t[j]
            return score

        def shard_xattn_Text_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1
            d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = torch.from_numpy(images[im_start:im_end])
                    s = torch.from_numpy(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end]
                    sim_list = model.xattn_score_Text_IMRAM(im_emb, s, l)
                    assert len(sim_list) == iteration_step
                    for k in range(iteration_step):
                        d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()
            score = 0
            for j in range(iteration_step):
                score += d[j]
            return score

        def shard_xattn_Image_IMRAM(model, images, captions, caplens, iteration_step, shard_size=128):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            n_im_shard = int((len(images) - 1) / shard_size) + 1
            n_cap_shard = int((len(captions) - 1) / shard_size) + 1
            None
            d = [np.zeros((len(images), len(captions))) for _ in range(iteration_step)]
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im_emb = torch.from_numpy(images[im_start:im_end])
                    s = torch.from_numpy(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end]
                    sim_list = model.xattn_score_Image_IMRAM(im_emb, s, l)
                    assert len(sim_list) == iteration_step
                    for k in range(iteration_step):
                        if len(sim_list[k]) != 0:
                            d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()
            score = 0
            for j in range(iteration_step):
                score += d[j]
            return score
        model_mode = kwargs.get('model_mode')
        iteration_step = kwargs.get('iteration_step')
        if model_mode == 'full_IMRAM':
            sims = shard_xattn_Full_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        elif model_mode == 'image_IMRAM':
            sims = shard_xattn_Image_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        elif model_mode == 'text_IMRAM':
            sims = shard_xattn_Text_IMRAM(model, img_embs, cap_embs, cap_lens, iteration_step)
        else:
            assert False, 'wrong model mode'
        return sims


def xattn_score_i2t(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, raw_feature_norm, smooth=lambda_softmax)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError('unknown aggfunc: {}'.format(agg_func))
        similarities.append(row_sim)
    similarities = torch.cat(similarities, 1)
    return similarities


def xattn_score_t2i(images, captions, cap_lens, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, **kwargs):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm, smooth=lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if agg_func == 'LogSumExp':
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / lambda_lse
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError('unknown aggfunc: {}'.format(agg_func))
        similarities.append(row_sim)
    similarities = torch.cat(similarities, 1)
    return similarities


class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, model_name, embed_size, vocab_size, word_dim, num_layers, image_dim, margin, max_violation, cross_attn, raw_feature_norm, agg_func, lambda_lse, lambda_softmax, use_bi_gru=True, image_norm=True, text_norm=True, **kwargs):
        super(SCAN, self).__init__()
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.cross_attn = cross_attn
        self.raw_feature_norm = raw_feature_norm
        self.agg_func = agg_func
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = lengths.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = lengths.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        if self.cross_attn == 't2i':
            scores = xattn_score_t2i(img_emb, cap_emb, cap_lens, raw_feature_norm=self.raw_feature_norm, agg_func=self.agg_func, lambda_lse=self.lambda_lse, lambda_softmax=self.lambda_softmax)
        elif self.cross_attn == 'i2t':
            scores = xattn_score_i2t(img_emb, cap_emb, cap_lens, raw_feature_norm=self.raw_feature_norm, agg_func=self.agg_func, lambda_lse=self.lambda_lse, lambda_softmax=self.lambda_softmax)
        else:
            raise ValueError('unknown attention type')
        loss = self.criterion(scores)
        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):

        def shard_xattn_t2i(images, captions, caplens, **kwargs):
            """
            Computer pairwise t2i image-caption distance with locality sharding
            """
            shard_size = kwargs.get('shard_size', 1000)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)
            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = torch.FloatTensor(images[im_start:im_end])
                    s = torch.FloatTensor(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end].tolist()
                    if torch.cuda.is_available():
                        im = im
                        s = s
                    sim = xattn_score_t2i(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = sim.cpu().detach().numpy()
            return d

        def shard_xattn_i2t(images, captions, caplens, **kwargs):
            """
            Computer pairwise i2t image-caption distance with locality sharding
            """
            shard_size = kwargs.get('shard_size', 1000)
            n_im_shard = int((len(images) - 1) / shard_size + 1)
            n_cap_shard = int((len(captions) - 1) / shard_size + 1)
            d = np.zeros((len(images), len(captions)))
            for i in range(n_im_shard):
                im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
                for j in range(n_cap_shard):
                    cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                    im = torch.FloatTensor(images[im_start:im_end])
                    s = torch.FloatTensor(captions[cap_start:cap_end])
                    l = caplens[cap_start:cap_end].tolist()
                    if torch.cuda.is_available():
                        im = im
                        s = s
                    sim = xattn_score_i2t(im, s, l, **kwargs)
                    d[im_start:im_end, cap_start:cap_end] = sim.cpu().detach().numpy()
            return d
        cross_attn = kwargs.get('cross_attn')
        if cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, **kwargs)
        elif cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, **kwargs)
        else:
            assert False, 'wrong cross attn'
        return sims


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(num_region), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)
        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)
        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()
        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)
        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-08):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = F.softmax(attn * smooth, dim=2)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)
    return weightedContext


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name='AVE', sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name
        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList([GraphReasoning(sim_dim) for i in range(sgr_step)])
        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError('Invalid input of opt.module_name in opts.py')
        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)
        sim_all = torch.cat(sim_all, 1)
        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SGRAF(nn.Module):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, model_name, module_name, sgr_step, embed_size, sim_dim, vocab_size, word_dim, num_layers, image_dim, margin, max_violation, use_bi_gru=True, image_norm=True, text_norm=True, **kwargs):
        super(SGRAF, self).__init__()
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.sim_enc = EncoderSimilarity(embed_size, sim_dim, module_name, sgr_step)
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)

    def forward_emb(self, batch):
        """Compute the image and caption embeddings"""
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        lengths = lengths.tolist()
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, batch):
        img_embs, cap_embs, cap_lens = batch
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        lengths = lengths.tolist()
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        sims = self.sim_enc(img_embs, cap_embs, lengths)
        loss = self.criterion(sims)
        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        shard_size = kwargs.get('shard_size', 100)
        n_im_shard = (len(img_embs) - 1) // shard_size + 1
        n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
        sims = np.zeros((len(img_embs), len(cap_embs)))
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))
                with torch.no_grad():
                    im = torch.FloatTensor(img_embs[im_start:im_end])
                    ca = torch.FloatTensor(cap_embs[ca_start:ca_end])
                    l = cap_lens[ca_start:ca_end].tolist()
                    if torch.cuda.is_available():
                        im = im
                        ca = ca
                    sim = model.forward_sim((im, ca, l))
                sims[im_start:im_end, ca_start:ca_end] = sim.cpu().detach().numpy()
        return sims


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class VSEPP(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, model_name, embed_size, vocab_size, word_dim, num_layers, finetune, cnn_type, margin, max_violation, use_bi_gru, measure, image_norm=True, text_norm=True, **kwargs):
        super(VSEPP, self).__init__()
        image_dim = None
        self.img_enc = EncoderImage(model_name, image_dim, embed_size, image_norm, cnn_type=cnn_type, finetune=finetune)
        self.txt_enc = EncoderText(model_name, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=use_bi_gru, text_norm=text_norm)
        self.criterion = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.mesure = measure

    def forward_emb(self, batch):
        """Compute the image and caption embeddings
        """
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = lengths.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        return img_emb, cap_emb, cap_lens

    def forward(self, batch):
        images = batch['image_feat']
        captions = batch['text_token']
        lengths = batch['text_len']
        if torch.cuda.is_available():
            images = images
            captions = captions
        cap_lens = lengths.tolist()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, cap_lens)
        if self.mesure == 'order':
            scores = order_sim(img_emb, cap_emb)
        else:
            scores = cosine_sim(img_emb, cap_emb)
        loss = self.criterion(scores)
        return loss

    @staticmethod
    def cal_sim(model, img_embs, cap_embs, cap_lens, **kwargs):
        sims = img_embs.dot(cap_embs.T)
        return sims


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AoA_Refiner_Core,
     lambda: ([], {'num_heads': 4, 'rnn_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (AttentionFiltration,
     lambda: ([], {'sim_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertBiAttention,
     lambda: ([], {'config': _mock_config(bi_hidden_size=4, bi_num_attention_heads=4, v_hidden_size=4, v_attention_probs_dropout_prob=0.5, hidden_size=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertBiOutput,
     lambda: ([], {'config': _mock_config(bi_hidden_size=4, v_hidden_size=4, v_hidden_dropout_prob=0.5, hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertImageAttention,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, v_num_attention_heads=4, dynamic_attention=4, hidden_size=4, v_attention_probs_dropout_prob=0.5, v_hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertImageIntermediate,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, v_intermediate_size=4, v_hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertImageOutput,
     lambda: ([], {'config': _mock_config(v_intermediate_size=4, v_hidden_size=4, v_hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertImagePooler,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, bi_hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertImageSelfAttention,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, v_num_attention_heads=4, dynamic_attention=4, hidden_size=4, v_attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertImageSelfOutput,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, v_hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertImgPredictionHeadTransform,
     lambda: ([], {'config': _mock_config(v_hidden_size=4, hidden_act=4, v_hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPredictionHeadTransform,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertTextPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4, bi_hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'vocab_size': 4, 'encoder_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (EncoderImageFull,
     lambda: ([], {'img_dim': 4, 'embed_size': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EncoderImagePrecomp,
     lambda: ([], {'img_dim': 4, 'embed_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fcn,
     lambda: ([], {'input_dim': 4, 'feature_dim': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GraphReasoning,
     lambda: ([], {'sim_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedDotAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleClassifier,
     lambda: ([], {'in_dim': 4, 'hid_dim': 4, 'out_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (TextSA,
     lambda: ([], {'embed_dim': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (VisualSA,
     lambda: ([], {'embed_dim': 4, 'dropout_rate': 0.5, 'num_region': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_njustkmg_OMML(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

