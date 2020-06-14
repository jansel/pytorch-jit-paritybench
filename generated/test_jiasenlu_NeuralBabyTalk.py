import sys
_module = sys.modules[__name__]
del sys
demo = _module
generate_robust_split = _module
main = _module
AttModel = _module
CaptionModel = _module
misc = _module
bak = _module
bbox_transform = _module
dataloader_coco = _module
dataloader_flickr30k = _module
dataloader_hdf = _module
eval_utils = _module
model = _module
resnet = _module
rewards = _module
utils = _module
vgg16 = _module
opts = _module
pooling = _module
roi_align = _module
_ext = _module
build = _module
functions = _module
modules = _module
roi_align = _module
prepro_det = _module
prepro_dic_coco = _module
prepro_dic_flickr = _module
prepro_ngrams = _module
prepro_ngrams_bak = _module
prepro_ngrams_flickr30k = _module
PyDataFormat = _module
jsonify_refs = _module
loadData = _module
cidereval = _module
pyciderevalcap = _module
cider = _module
cider_scorer = _module
ciderD = _module
ciderD = _module
ciderD_scorer = _module
eval = _module
tokenizer = _module
ptbtokenizer = _module
sentence_gen_tools = _module
coco_eval = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


import numpy as np


import time


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


from torch.autograd import *


from torch.nn.parameter import Parameter


import math


import random


import string


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import collections


import types


import warnings


import itertools


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import copy


from collections import defaultdict


class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -100000000.0

    def forward(self, h, att_feats, p_att_feats):
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


class Attention2(nn.Module):

    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -100000000.0

    def forward(self, h, att_feats, p_att_feats, mask):
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        hAflat = self.alpha_net(dot)
        hAflat = hAflat.view(-1, att_size)
        hAflat.masked_fill_(mask, self.min_value)
        weight = F.softmax(hAflat, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


class adaPnt(nn.Module):

    def __init__(self, conv_size, rnn_size, att_hid_size, dropout,
        min_value, beta):
        super(adaPnt, self).__init__()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_hid_size = att_hid_size
        self.min_value = min_value
        self.conv_size = conv_size
        self.f_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.f_fc2 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.h_fc1 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.inplace = False
        self.beta = beta

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, mask):
        batch_size = h_out.size(0)
        roi_num = conv_feat_embed.size(1)
        conv_feat_embed = conv_feat_embed.view(-1, roi_num, self.att_hid_size)
        fake_region = F.relu(self.f_fc1(fake_region.view(-1, self.rnn_size)
            ), inplace=self.inplace)
        fake_region_embed = self.f_fc2(fake_region)
        h_out_embed = self.h_fc1(h_out)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.
            att_hid_size), conv_feat_embed], 1)
        hA = F.tanh(img_all_embed + h_out_embed.view(-1, 1, self.att_hid_size))
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        hAflat = hAflat.view(-1, roi_num + 1)
        hAflat.masked_fill_(mask, self.min_value)
        return hAflat


class TopDownCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -100000000.0
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size,
            opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)
        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.
            att_hid_size, self.drop_prob_lm, self.min_value, opt.beta)
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats,
        p_pool_feats, att_mask, pnt_mask, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0])
            )
        att = self.attention(h_att, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att, pool_feats, p_pool_feats, att_mask[:, 1:]
            )
        lang_lstm_input = torch.cat([att + att2, h_att], 1)
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2
            (state[0][1]))
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1],
            state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        fake_box = F.dropout(ada_gate_point * F.tanh(state[1][1]), self.
            drop_prob_lm, training=self.training)
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats,
            pnt_mask)
        state = torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])
        return output, det_prob, state


class Att2in2Core(nn.Module):

    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.min_value = -100000000.0
        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.
            att_hid_size, self.drop_prob_lm, self.min_value, opt.beta)
        self.a2c1 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.a2c2 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 6 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)
        self.dropout1 = nn.Dropout(self.drop_prob_lm)
        self.dropout2 = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, pool_feats,
        p_pool_feats, att_mask, pnt_mask, state):
        att_res1 = self.attention(state[0][-1], att_feats, p_att_feats)
        att_res2 = self.attention2(state[0][-1], pool_feats, p_pool_feats,
            att_mask[:, 1:])
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 4 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        s_gate = sigmoid_chunk.narrow(1, self.rnn_size * 3, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 4 * self.rnn_size, 2 * self
            .rnn_size) + self.a2c1(att_res1) + self.a2c2(att_res2)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        fake_box = s_gate * F.tanh(next_c)
        output = self.dropout1(next_h)
        fake_box = self.dropout2(fake_box)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats,
            pnt_mask)
        return output, det_prob, state


class CascadeCore(nn.Module):

    def __init__(self, opt):
        super(CascadeCore, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fg_size = opt.fg_size
        self.fg_size = opt.fg_size
        self.bn_fc = nn.Sequential(nn.Linear(opt.rnn_size + opt.rnn_size,
            opt.rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm), nn.
            Linear(opt.rnn_size, 2))
        self.fg_fc = nn.Sequential(nn.Linear(opt.rnn_size + opt.rnn_size,
            opt.rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm), nn.
            Linear(opt.rnn_size, 300))
        self.fg_emb = Parameter(opt.glove_fg)
        self.fg_emb.requires_grad = False
        self.fg_mask = Parameter(opt.fg_mask)
        self.fg_mask.requires_grad = False
        self.min_value = -100000000.0
        self.beta = opt.beta

    def forward(self, fg_idx, pool_feats, rnn_outs, roi_labels,
        seq_batch_size, seq_cnt):
        roi_num = pool_feats.size(1)
        pool_feats = pool_feats.view(seq_batch_size, 1, roi_num, self.rnn_size
            ) * roi_labels.view(seq_batch_size, seq_cnt, roi_num, 1)
        pool_cnt = roi_labels.sum(2)
        pool_cnt[pool_cnt == 0] = 1
        pool_feats = pool_feats.sum(2) / pool_cnt.view(seq_batch_size,
            seq_cnt, 1)
        pool_feats = torch.cat((rnn_outs, pool_feats), 2)
        bn_logprob = F.log_softmax(self.bn_fc(pool_feats), dim=2)
        fg_out = self.fg_fc(pool_feats)
        fg_score = torch.mm(fg_out.view(-1, 300), self.fg_emb.t()).view(
            seq_batch_size, -1, self.fg_size + 1)
        fg_mask = self.fg_mask[fg_idx]
        fg_score.masked_fill_(fg_mask.view_as(fg_score), self.min_value)
        fg_logprob = F.log_softmax(self.beta * fg_score, dim=2)
        return bn_logprob, fg_logprob


class CaptionModel(nn.Module):

    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, rnn_output, det_prob, beam_fc_feats,
        beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
        beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset,
        opt):

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs,
            beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs,
            beam_fg_seq, beam_fg_seq_logprobs, rnn_output, beam_pnt_mask, state
            ):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p':
                        candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()
                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, (vix)] = beam_seq_prev[:, (v['q'])]
                    beam_seq_logprobs[:t, (vix)] = beam_seq_logprobs_prev[:,
                        (v['q'])]
                    beam_bn_seq[:t, (vix)] = beam_bn_seq_prev[:, (v['q'])]
                    beam_bn_seq_logprobs[:t, (vix)
                        ] = beam_bn_seq_logprobs_prev[:, (v['q'])]
                    beam_fg_seq[:t, (vix)] = beam_fg_seq_prev[:, (v['q'])]
                    beam_fg_seq_logprobs[:t, (vix)
                        ] = beam_fg_seq_logprobs_prev[:, (v['q'])]
                    beam_pnt_mask[:, (vix)] = beam_pnt_mask_prev[:, (v['q'])]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, (vix)] = state[state_ix][:, (v['q'])
                        ]
                new_rnn_output[vix] = rnn_output[v['q']]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            rnn_output = new_rnn_output
            return (beam_seq, beam_seq_logprobs, beam_logprobs_sum,
                beam_bn_seq, beam_bn_seq_logprobs, beam_fg_seq,
                beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(),
                candidates)
        beam_size = opt.get('beam_size', 5)
        beam_att_mask = beam_pnt_mask.clone()
        rois_num = beam_ppls.size(1)
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)
        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            det_prob = F.log_softmax(det_prob, dim=1)
            decoded = F.log_softmax(self.logit(rnn_output), dim=1)
            lambda_v = det_prob[:, (0)].contiguous()
            prob_det = det_prob[:, 1:].contiguous()
            decoded = decoded + lambda_v.view(beam_size, 1).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)
            logprobsf = logprobs.data.cpu()
            (beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, state, beam_pnt_mask_new, candidates_divm) = (
                beam_step(logprobsf, beam_size, t, beam_seq,
                beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, beam_pnt_mask_list[-1].t(), state))
            it = beam_seq[t]
            roi_idx = it.clone() - self.vocab_size - 1
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset
            roi_idx_offset[roi_mask] = 0
            vis_idx = beam_ppls.data[:, :, (4)].contiguous().view(-1)[
                roi_idx_offset].long()
            vis_idx[roi_mask] = 0
            it_new = it.clone()
            it_new[it > self.vocab_size] = vis_idx[roi_mask == 0
                ] + self.vocab_size
            roi_labels = beam_pool_feats.data.new(beam_size * rois_num).zero_()
            if (roi_mask == 0).sum() > 0:
                roi_labels[roi_idx_offset[roi_mask == 0]] = 1
            roi_labels = roi_labels.view(beam_size, 1, rois_num)
            bn_logprob, fg_logprob = self.ccr_core(vis_idx, beam_pool_feats,
                rnn_output.view(beam_size, 1, self.rnn_size), Variable(
                roi_labels), beam_size, 1)
            bn_logprob = bn_logprob.view(beam_size, -1)
            fg_logprob = fg_logprob.view(beam_size, -1)
            slp_bn, it_bn = torch.max(bn_logprob.data, 1)
            slp_fg, it_fg = torch.max(fg_logprob.data, 1)
            it_bn[roi_mask] = 0
            it_fg[roi_mask] = 0
            beam_bn_seq[t] = it_bn
            beam_bn_seq_logprobs[t] = slp_bn
            beam_fg_seq[t] = it_fg
            beam_fg_seq_logprobs[t] = slp_fg
            for vix in range(beam_size):
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {'seq': beam_seq[:, (vix)].clone(),
                        'logps': beam_seq_logprobs[:, (vix)].clone(), 'p':
                        beam_logprobs_sum[vix], 'bn_seq': beam_bn_seq[:, (
                        vix)].clone(), 'bn_logps': beam_bn_seq_logprobs[:,
                        (vix)].clone(), 'fg_seq': beam_fg_seq[:, (vix)].
                        clone(), 'fg_logps': beam_fg_seq_logprobs[:, (vix)]
                        .clone()}
                    done_beams.append(final_beam)
                    beam_logprobs_sum[vix] = -1000
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            beam_pnt_mask = beam_pnt_mask_new.data.clone()
            beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
            beam_pnt_mask.view(-1)[0] = 0
            beam_pnt_mask_list.append(Variable(beam_pnt_mask))
            xt = self.embed(Variable(it_new))
            rnn_output, det_prob, state = self.core(xt, beam_fc_feats,
                beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
                beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], state
                )
        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def constraint_beam_search(self, state, rnn_output, det_prob,
        beam_fc_feats, beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
        beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset,
        tag_size, tags, opt):
        """
        Implementation of the constraint beam search for image captioning.
        """

        def constraint_beam_step(logprobsf, beam_size, t, beam_seq,
            beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
            beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
            rnn_output, beam_pnt_mask, tags, vocab_size, tag_num, state):
            tag_list = range(vocab_size + 1, vocab_size + tag_num + 1)
            ys, ix = torch.sort(logprobsf, 2, True)
            candidates = {tag: [] for tag in tags}
            if t == 0:
                num_fsm = 1
            else:
                num_fsm = len(tags)
            for s in range(num_fsm):
                cols = min(beam_size, ys.size(2))
                if t == 0:
                    rows = 1
                else:
                    rows = torch.sum(beam_seq[0][s] != 0)
                for q in range(rows):
                    tagSet = utils.containSet(tags, tags[s])
                    for tag in tagSet:
                        if tag == tags[s]:
                            tmpIdx = []
                            ii = 0
                            while len(tmpIdx) < cols:
                                if ix[s, q, ii] not in tag_list:
                                    tmpIdx.append(ii)
                                ii += 1
                            for c in range(cols):
                                local_logprob = ys[s, q, tmpIdx[c]]
                                cc = ix[s, q, tmpIdx[c]]
                                candidate_logprob = beam_logprobs_sum[s, q
                                    ] + local_logprob
                                candidates[tag].append({'c': cc, 'q': q,
                                    'p': candidate_logprob, 'r':
                                    local_logprob, 's': s})
                        else:
                            tag_diff = set(tag) - set(tags[s])
                            for tag_idx in tag_diff:
                                local_logprob = logprobsf[s, q, tag_idx +
                                    vocab_size + 1]
                                cc = tag_idx + vocab_size + 1
                            candidate_logprob = beam_logprobs_sum[s, q
                                ] + local_logprob
                            candidates[tag].append({'c': cc, 'q': q, 'p':
                                candidate_logprob, 'r': local_logprob, 's': s})
            for tag, candidate in candidates.items():
                candidates[tag] = sorted(candidate, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()
                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()
            for s in range(len(tags)):
                tag = tags[s]
                v_tag = candidates[tag]
                for vix in range(min(len(v_tag), beam_size)):
                    v = v_tag[vix]
                    if t >= 1:
                        beam_seq[:t, (s), (vix)] = beam_seq_prev[:, (v['s']
                            ), (v['q'])]
                        beam_seq_logprobs[:t, (s), (vix)
                            ] = beam_seq_logprobs_prev[:, (v['s']), (v['q'])]
                        beam_bn_seq[:t, (s), (vix)] = beam_bn_seq_prev[:, (
                            v['s']), (v['q'])]
                        beam_bn_seq_logprobs[:t, (s), (vix)
                            ] = beam_bn_seq_logprobs_prev[:, (v['s']), (v['q'])
                            ]
                        beam_fg_seq[:t, (s), (vix)] = beam_fg_seq_prev[:, (
                            v['s']), (v['q'])]
                        beam_fg_seq_logprobs[:t, (s), (vix)
                            ] = beam_fg_seq_logprobs_prev[:, (v['s']), (v['q'])
                            ]
                        beam_pnt_mask[:, (vix)] = beam_pnt_mask_prev[:, (v[
                            'q'])]
                    for state_ix in range(len(new_state)):
                        new_state[state_ix][:, (vix + s * beam_size)] = state[
                            state_ix][:, (v['q'] + v['s'] * beam_size)]
                    new_rnn_output[vix + s * beam_size] = rnn_output[v['q'] +
                        v['s'] * beam_size]
                    beam_seq[t, s, vix] = v['c']
                    beam_seq_logprobs[t, s, vix] = v['r']
                    beam_logprobs_sum[s, vix] = v['p']
            state = new_state
            rnn_output = new_rnn_output
            return (beam_seq, beam_seq_logprobs, beam_logprobs_sum,
                beam_bn_seq, beam_bn_seq_logprobs, beam_fg_seq,
                beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(),
                candidates)
        beam_size = int(opt.get('beam_size', 5) / len(tags))
        num_fsm = len(tags)
        rois_num = beam_ppls.size(1)
        beam_att_mask = beam_pnt_mask.clone()
        beam_seq = torch.LongTensor(self.seq_length, num_fsm, beam_size).zero_(
            )
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, num_fsm,
            beam_size).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, num_fsm, beam_size
            ).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, num_fsm,
            beam_size).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, num_fsm, beam_size
            ).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, num_fsm,
            beam_size).zero_()
        beam_logprobs_sum = torch.zeros(num_fsm, beam_size)
        done_beams = {tag: [] for tag in tags}
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)
        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            det_prob = F.log_softmax(det_prob, dim=1)
            decoded = F.log_softmax(self.logit(rnn_output), dim=1)
            lambda_v = det_prob[:, (0)].contiguous()
            prob_det = det_prob[:, 1:].contiguous()
            decoded = decoded + lambda_v.view(beam_size * num_fsm, 1
                ).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)
            logprobsf = logprobs.view(num_fsm, beam_size, -1).data.cpu()
            (beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, state, beam_pnt_mask_new, candidates_divm) = (
                constraint_beam_step(logprobsf, beam_size, t, beam_seq,
                beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, beam_pnt_mask_list[-1].t(), tags, self.
                vocab_size, tag_size, state))
            it = beam_seq[t].view(-1)
            roi_idx = it.clone() - self.vocab_size - 1
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset
            roi_idx_offset[roi_mask] = 0
            vis_idx = beam_ppls.data[:, :, (4)].contiguous().view(-1)[
                roi_idx_offset].long()
            vis_idx[roi_mask] = 0
            it_new = it.clone()
            it_new[it > self.vocab_size] = vis_idx[roi_mask == 0
                ] + self.vocab_size
            roi_labels = beam_pool_feats.data.new(beam_size * num_fsm *
                rois_num).zero_()
            if (roi_mask == 0).sum() > 0:
                roi_labels[roi_idx_offset[roi_mask == 0]] = 1
            roi_labels = roi_labels.view(beam_size * num_fsm, 1, rois_num)
            bn_logprob, fg_logprob = self.ccr_core(vis_idx, beam_pool_feats,
                rnn_output.view(beam_size * num_fsm, 1, self.rnn_size),
                Variable(roi_labels), beam_size * num_fsm, 1)
            bn_logprob = bn_logprob.view(beam_size * num_fsm, -1)
            fg_logprob = fg_logprob.view(beam_size * num_fsm, -1)
            slp_bn, it_bn = torch.max(bn_logprob.data, 1)
            slp_fg, it_fg = torch.max(fg_logprob.data, 1)
            it_bn[roi_mask] = 0
            it_fg[roi_mask] = 0
            beam_bn_seq[t] = it_bn
            beam_bn_seq_logprobs[t] = slp_bn
            beam_fg_seq[t] = it_fg
            beam_fg_seq_logprobs[t] = slp_fg
            for s in range(num_fsm):
                for vix in range(beam_size):
                    if beam_seq[0, s, vix] != 0 and (beam_seq[t, s, vix] ==
                        0 or t == self.seq_length - 1):
                        final_beam = {'seq': beam_seq[:, (s), (vix)].clone(
                            ), 'logps': beam_seq_logprobs[:, (s), (vix)].
                            clone(), 'p': beam_logprobs_sum[s, vix],
                            'bn_seq': beam_bn_seq[:, (s), (vix)].clone(),
                            'bn_logps': beam_bn_seq_logprobs[:, (s), (vix)]
                            .clone(), 'fg_seq': beam_fg_seq[:, (s), (vix)].
                            clone(), 'fg_logps': beam_fg_seq_logprobs[:, (s
                            ), (vix)].clone()}
                        done_beams[tags[s]].append(final_beam)
                        beam_logprobs_sum[s, vix] = -1000
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            beam_pnt_mask = beam_pnt_mask_new.data.clone()
            beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
            beam_pnt_mask.view(-1)[0] = 0
            beam_pnt_mask_list.append(Variable(beam_pnt_mask))
            xt = self.embed(Variable(it_new))
            rnn_output, det_prob, state = self.core(xt, beam_fc_feats,
                beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
                beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], state
                )
        for tag, beams in done_beams.items():
            done_beams[tag] = sorted(beams, key=lambda x: -x['p'])
        return done_beams


class CaptionModel(nn.Module):

    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, rnn_output, det_prob, beam_fc_feats,
        beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
        beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset,
        opt):

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs,
            beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs,
            beam_fg_seq, beam_fg_seq_logprobs, rnn_output, beam_pnt_mask, state
            ):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p':
                        candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()
                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, (vix)] = beam_seq_prev[:, (v['q'])]
                    beam_seq_logprobs[:t, (vix)] = beam_seq_logprobs_prev[:,
                        (v['q'])]
                    beam_bn_seq[:t, (vix)] = beam_bn_seq_prev[:, (v['q'])]
                    beam_bn_seq_logprobs[:t, (vix)
                        ] = beam_bn_seq_logprobs_prev[:, (v['q'])]
                    beam_fg_seq[:t, (vix)] = beam_fg_seq_prev[:, (v['q'])]
                    beam_fg_seq_logprobs[:t, (vix)
                        ] = beam_fg_seq_logprobs_prev[:, (v['q'])]
                    beam_pnt_mask[:, (vix)] = beam_pnt_mask_prev[:, (v['q'])]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, (vix)] = state[state_ix][:, (v['q'])
                        ]
                new_rnn_output[vix] = rnn_output[v['q']]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            rnn_output = new_rnn_output
            return (beam_seq, beam_seq_logprobs, beam_logprobs_sum,
                beam_bn_seq, beam_bn_seq_logprobs, beam_fg_seq,
                beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(),
                candidates)
        beam_size = opt.get('beam_size', 5)
        beam_att_mask = beam_pnt_mask.clone()
        rois_num = beam_ppls.size(1)
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size
            ).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)
        logprobsf = {}
        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            det_prob = F.log_softmax(det_prob, dim=1)
            decoded = F.log_softmax(self.logit(rnn_output), dim=1)
            lambda_v = det_prob[:, (0)].contiguous()
            prob_det = det_prob[:, 1:].contiguous()
            decoded = decoded + lambda_v.view(beam_size, 1).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)
            logprobsf[tag] = logprobs.data.cpu()
            (beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, state, beam_pnt_mask_new, candidates_divm) = (
                beam_step(logprobsf, beam_size, t, beam_seq,
                beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
                beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
                rnn_output, beam_pnt_mask_list[-1].t(), state))
            it = beam_seq[t]
            roi_idx = it.clone() - self.vocab_size - 1
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset
            roi_idx_offset[roi_mask] = 0
            vis_idx = beam_ppls.data[:, :, (4)].contiguous().view(-1)[
                roi_idx_offset].long()
            vis_idx[roi_mask] = 0
            it_new = it.clone()
            it_new[it > self.vocab_size] = vis_idx[roi_mask == 0
                ] + self.vocab_size
            roi_labels = beam_pool_feats.data.new(beam_size * rois_num).zero_()
            if (roi_mask == 0).sum() > 0:
                roi_labels[roi_idx_offset[roi_mask == 0]] = 1
            roi_labels = roi_labels.view(beam_size, 1, rois_num)
            bn_logprob, fg_logprob = self.ccr_core(vis_idx, beam_pool_feats,
                rnn_output.view(beam_size, 1, self.rnn_size), Variable(
                roi_labels), beam_size, 1)
            bn_logprob = bn_logprob.view(beam_size, -1)
            fg_logprob = fg_logprob.view(beam_size, -1)
            slp_bn, it_bn = torch.max(bn_logprob.data, 1)
            slp_fg, it_fg = torch.max(fg_logprob.data, 1)
            it_bn[roi_mask] = 0
            it_fg[roi_mask] = 0
            beam_bn_seq[t] = it_bn
            beam_bn_seq_logprobs[t] = slp_bn
            beam_fg_seq[t] = it_fg
            beam_fg_seq_logprobs[t] = slp_fg
            for vix in range(beam_size):
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {'seq': beam_seq[:, (vix)].clone(),
                        'logps': beam_seq_logprobs[:, (vix)].clone(), 'p':
                        beam_logprobs_sum[vix], 'bn_seq': beam_bn_seq[:, (
                        vix)].clone(), 'bn_logps': beam_bn_seq_logprobs[:,
                        (vix)].clone(), 'fg_seq': beam_fg_seq[:, (vix)].
                        clone(), 'fg_logps': beam_fg_seq_logprobs[:, (vix)]
                        .clone()}
                    done_beams.append(final_beam)
                    beam_logprobs_sum[vix] = -1000
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            beam_pnt_mask = beam_pnt_mask_new.data.clone()
            beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
            beam_pnt_mask.view(-1)[0] = 0
            beam_pnt_mask_list.append(Variable(beam_pnt_mask))
            xt = self.embed(Variable(it_new))
            rnn_output, det_prob, state = self.core(xt, beam_fc_feats,
                beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
                beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], state
                )
        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def constraint_beam_search(self, state, rnn_output, det_prob,
        beam_fc_feats, beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
        beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset,
        tag_size, tags, opt):
        """
        Implementation of the constraint beam search for image captioning.
        """

        def constraint_beam_step(logprobsf, beam_size, t, beam_seq,
            beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
            beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs,
            rnn_output, beam_pnt_mask, tags, vocab_size, tag_num, state):
            tag_list = range(vocab_size + 1, vocab_size + tag_num + 1)
            if t == 0:
                for keys, logprobs in logprobsf.items():
                    logprobsf[keys] = logprobs.view(1, beam_size, -1)
            candidates = {tag: [] for tag in tags}
            if t == 0:
                num_fsm = 1
            else:
                num_fsm = len(tags)
            for s in range(num_fsm):
                ys, ix = torch.sort(logprobsf[tags[s]], 2, True)
                cols = min(beam_size, ys.size(2))
                if t == 0:
                    rows = 1
                else:
                    rows = torch.sum(beam_seq[0][s] != 0)
                for q in range(rows):
                    tagSet = utils.containSet(tags, tags[q])
                    for tag in tagSet:
                        if tag == tags[q]:
                            tmpIdx = []
                            ii = 0
                            while len(tmpIdx) < cols:
                                if ix[s, q, ii] not in tag_list:
                                    tmpIdx.append(ii)
                                    ii += 1
                            for c in range(cols):
                                local_logprob = ys[s, q, tmpIdx[c]]
                                cc = ix[s, q, tmpIdx[c]]
                                candidate_logprob = beam_logprobs_sum[s, q
                                    ] + local_logprob
                                candidates[tag].append({'c': cc, 'q': q,
                                    'p': candidate_logprob, 'r': local_logprob}
                                    )
                        else:
                            tag_diff = set(tag) - set(tags[q])
                            for tag_idx in tag_diff:
                                local_logprob = logprobsf[s, q, tag_idx +
                                    vocab_size + 1]
                                cc = tag_idx + vocab_size + 1
                            candidate_logprob = beam_logprobs_sum[s, q
                                ] + local_logprob
                            candidates[tag].append({'c': cc, 'q': q, 'p':
                                candidate_logprob, 'r': local_logprob})
            for tag, candidate in candidates.items():
                candidates[tag] = sorted(candidate, key=lambda x: -x['p'])
            pdb.set_trace()
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()
                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()
            for vix, tag in enumerate(tags):
                v = candidates[tag]
                pdb.set_trace()
                if len(v) != 0:
                    v = v[0]
                    if t >= 1:
                        beam_seq[:t, (vix)] = beam_seq_prev[:, (v['q'])]
                        beam_seq_logprobs[:t, (vix)] = beam_seq_logprobs_prev[:
                            , (v['q'])]
                        beam_bn_seq[:t, (vix)] = beam_bn_seq_prev[:, (v['q'])]
                        beam_bn_seq_logprobs[:t, (vix)
                            ] = beam_bn_seq_logprobs_prev[:, (v['q'])]
                        beam_fg_seq[:t, (vix)] = beam_fg_seq_prev[:, (v['q'])]
                        beam_fg_seq_logprobs[:t, (vix)
                            ] = beam_fg_seq_logprobs_prev[:, (v['q'])]
                        beam_pnt_mask[:, (vix)] = beam_pnt_mask_prev[:, (v[
                            'q'])]
                    for state_ix in range(len(new_state)):
                        new_state[state_ix][:, (vix)] = state[state_ix][:,
                            (v['q'])]
                    new_rnn_output[vix] = rnn_output[v['q']]
                    beam_seq[t, vix] = v['c']
                    beam_seq_logprobs[t, vix] = v['r']
                    beam_logprobs_sum[vix] = v['p']
            state = new_state
            rnn_output = new_rnn_output
            return (beam_seq, beam_seq_logprobs, beam_logprobs_sum,
                beam_bn_seq, beam_bn_seq_logprobs, beam_fg_seq,
                beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(),
                candidates)
        beam_size = opt.get('beam_size', 5)
        rois_num = beam_ppls.size(1)
        beam_att_mask = beam_pnt_mask.clone()
        fsm_num = len(tags)
        beam_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size).zero_(
            )
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num,
            beam_size).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size
            ).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num,
            beam_size).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size
            ).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num,
            beam_size).zero_()
        beam_logprobs_sum = torch.zeros(fsm_num, beam_size)
        done_beams = {tag: [] for tag in tags}
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)
        state = {(): state}
        det_prob = {(): det_prob}
        rnn_output = {(): rnn_output}
        logprobsf = {}
        for t in range(self.seq_length):
            for tag in state.keys():
                """pem a beam merge. that is,
                for every previous beam we now many new possibilities to branch out
                we need to resort our beams to maintain the loop invariant of keeping
                the top beam_size most likely sequences."""
                det_prob[tag] = F.log_softmax(det_prob[tag], dim=1)
                decoded = F.log_softmax(self.logit(rnn_output[tag]), dim=1)
                lambda_v = det_prob[tag][:, (0)].contiguous()
                prob_det = det_prob[tag][:, 1:].contiguous()
                decoded = decoded + lambda_v.view(beam_size, 1).expand_as(
                    decoded)
                logprobs = torch.cat([decoded, prob_det], 1)
                logprobsf[tag] = logprobs.data.cpu()
                (beam_seq, beam_seq_logprobs, beam_logprobs_sum,
                    beam_bn_seq, beam_bn_seq_logprobs, beam_fg_seq,
                    beam_fg_seq_logprobs, rnn_output, state,
                    beam_pnt_mask_new, candidates_divm) = (constraint_beam_step
                    (logprobsf, beam_size, t, beam_seq, beam_seq_logprobs,
                    beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs,
                    beam_fg_seq, beam_fg_seq_logprobs, rnn_output,
                    beam_pnt_mask_list[-1].t(), tags, self.vocab_size,
                    tag_size, state))
                it = beam_seq[t]
                roi_idx = it.clone() - self.vocab_size - 1
                roi_mask = roi_idx < 0
                roi_idx_offset = roi_idx + vis_offset
                roi_idx_offset[roi_mask] = 0
                vis_idx = beam_ppls.data[:, :, (4)].contiguous().view(-1)[
                    roi_idx_offset].long()
                vis_idx[roi_mask] = 0
                it_new = it.clone()
                it_new[it > self.vocab_size] = vis_idx[roi_mask == 0
                    ] + self.vocab_size
                roi_labels = beam_pool_feats.data.new(beam_size * rois_num
                    ).zero_()
                if (roi_mask == 0).sum() > 0:
                    roi_labels[roi_idx_offset[roi_mask == 0]] = 1
                roi_labels = roi_labels.view(beam_size, 1, rois_num)
                bn_logprob, fg_logprob = self.ccr_core(vis_idx,
                    beam_pool_feats, rnn_output.view(beam_size, 1, self.
                    rnn_size), Variable(roi_labels), beam_size, 1)
                bn_logprob = bn_logprob.view(beam_size, -1)
                fg_logprob = fg_logprob.view(beam_size, -1)
                slp_bn, it_bn = torch.max(bn_logprob.data, 1)
                slp_fg, it_fg = torch.max(fg_logprob.data, 1)
                it_bn[roi_mask] = 0
                it_fg[roi_mask] = 0
                beam_bn_seq[t] = it_bn
                beam_bn_seq_logprobs[t] = slp_bn
                beam_fg_seq[t] = it_fg
                beam_fg_seq_logprobs[t] = slp_fg
                for vix in range(beam_size):
                    if beam_seq[0, vix] != 0 and (beam_seq[t, vix] == 0 or 
                        t == self.seq_length - 1):
                        constraint_word = tags[vix]
                        skip_flag = False
                        for ii in constraint_word:
                            idx = ii + self.vocab_size + 1
                            idx_0 = 0
                            for jj in range(self.seq_length):
                                if beam_seq[jj, vix] == 0:
                                    idx_0 = jj
                                    break
                            if idx_0 == 0:
                                idx_0 = self.seq_length
                            if idx == beam_seq[idx_0 - 1, vix
                                ] or idx == beam_seq[idx_0 - 2, vix]:
                                skip_flag = True
                        if skip_flag == False:
                            final_beam = {'seq': beam_seq[:, (vix)].clone(),
                                'logps': beam_seq_logprobs[:, (vix)].clone(
                                ), 'p': beam_logprobs_sum[vix], 'bn_seq':
                                beam_bn_seq[:, (vix)].clone(), 'bn_logps':
                                beam_bn_seq_logprobs[:, (vix)].clone(),
                                'fg_seq': beam_fg_seq[:, (vix)].clone(),
                                'fg_logps': beam_fg_seq_logprobs[:, (vix)].
                                clone()}
                            done_beams[tags[vix]].append(final_beam)
                            beam_logprobs_sum[vix] = -1000
                pnt_idx_offset = roi_idx + roi_offset + 1
                pnt_idx_offset[roi_mask] = 0
                beam_pnt_mask = beam_pnt_mask_new.data.clone()
                beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
                beam_pnt_mask.view(-1)[0] = 0
                beam_pnt_mask_list.append(Variable(beam_pnt_mask))
                xt = self.embed(Variable(it_new))
                rnn_output, det_prob, state = self.core(xt, beam_fc_feats,
                    beam_conv_feats, beam_p_conv_feats, beam_pool_feats,
                    beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1
                    ], state)
        for tag, beams in done_beams.items():
            done_beams[tag] = sorted(beams, key=lambda x: -x['p'])
        return done_beams


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
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


model_urls = {'resnet18':
    'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth'}


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class resnet(nn.Module):

    def __init__(self, opt, _num_layers=101, _fixed_block=1, pretrained=True):
        super(resnet, self).__init__()
        self._num_layers = _num_layers
        self._fixed_block = _fixed_block
        self.pretrained = pretrained
        self.model_path = '%s/imagenet_weights/resnet' % opt.data_path + str(
            _num_layers) + '.pth'
        if self._num_layers == 50:
            self.resnet = resnet50(pretrained=False)
        elif self._num_layers == 101:
            self.resnet = resnet101(pretrained=False)
        elif self._num_layers == 152:
            self.resnet = resnet152(pretrained=False)
        else:
            raise NotImplementedError
        if self.pretrained == True:
            None
            state_dict = torch.load(self.model_path)
            self.resnet.load_state_dict({k: v for k, v in state_dict.items(
                ) if k in self.resnet.state_dict()})
        for p in self.resnet.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet.conv1.parameters():
            p.requires_grad = False
        assert 0 <= _fixed_block <= 4
        if _fixed_block >= 4:
            for p in self.resnet.layer4.parameters():
                p.requires_grad = False
        if _fixed_block >= 3:
            for p in self.resnet.layer3.parameters():
                p.requires_grad = False
        if _fixed_block >= 2:
            for p in self.resnet.layer2.parameters():
                p.requires_grad = False
        if _fixed_block >= 1:
            for p in self.resnet.layer1.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.resnet.apply(set_bn_fix)
        self.cnn_net = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
            self.resnet.relu, self.resnet.maxpool, self.resnet.layer1, self
            .resnet.layer2, self.resnet.layer3, self.resnet.layer4)

    def forward(self, img):
        conv_feat = self.cnn_net(img)
        fc_feat = conv_feat.mean(3).mean(2)
        return conv_feat, fc_feat

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.resnet.eval()
            if self._fixed_block <= 3:
                self.resnet.layer4.train()
            if self._fixed_block <= 2:
                self.resnet.layer3.train()
            if self._fixed_block <= 1:
                self.resnet.layer2.train()
            if self._fixed_block <= 0:
                self.resnet.layer1.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.resnet.apply(set_bn_eval)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


class get_self_critical_reward(nn.Module):

    def __init__(self, opt):
        super(get_self_critical_reward, self).__init__()
        self.vocab_size = opt.vocab_size
        self.st2towidx = opt.st2towidx
        self.opt = opt
        self.CiderD_scorer = CiderD(df=opt.cached_tokens)

    def forward(self, gen_input, greedy_input, gt_gts, ncap):
        gen_txt_seq, gen_bn_seq, gen_vis_seq = gen_input
        greedy_txt_seq, greedy_bn_seq, greedy_vis_seq = greedy_input
        self.st2towidx = self.st2towidx.type_as(gen_txt_seq)
        batch_size = gen_txt_seq.size(0)
        seq_per_img = batch_size // gt_gts.size(0)
        gen_result = gen_txt_seq.new(gen_txt_seq.size()).zero_()
        greedy_result = greedy_txt_seq.new(greedy_txt_seq.size()).zero_()
        gen_mask = gen_txt_seq < self.vocab_size
        gen_vis_seq = gen_vis_seq.view(batch_size, -1)
        gen_bn_seq = gen_bn_seq.view(batch_size, -1)
        gen_result[gen_mask] = gen_txt_seq[gen_mask]
        gen_vis_idx = gen_vis_seq[gen_mask == 0] * 2 + gen_bn_seq[gen_mask == 0
            ] - 1
        gen_result[gen_mask == 0] = self.st2towidx[gen_vis_idx]
        greedy_mask = greedy_txt_seq < self.vocab_size
        greedy_vis_seq = greedy_vis_seq.view(batch_size, -1)
        greedy_bn_seq = greedy_bn_seq.view(batch_size, -1)
        greedy_result[greedy_mask] = greedy_txt_seq[greedy_txt_seq < self.
            vocab_size]
        greedy_vis_idx = greedy_vis_seq[greedy_mask == 0] * 2 + greedy_bn_seq[
            greedy_mask == 0] - 1
        greedy_result[greedy_mask == 0] = self.st2towidx[greedy_vis_idx]
        res = OrderedDict()
        gen_result = gen_result.cpu().numpy()
        greedy_result = greedy_result.cpu().numpy()
        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i])]
        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_result[i])]
        gts = OrderedDict()
        for i in range(batch_size):
            gts_np = gt_gts[i][:ncap.data[i]].data.cpu().numpy()
            gts[i] = [array_to_str(gts_np[j]) for j in range(len(gts_np))]
        res = [{'image_id': i, 'caption': res[i]} for i in range(2 *
            batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 *
            batch_size)}
        _, scores = self.CiderD_scorer.compute_score(gts, res)
        scores = scores[:batch_size] - scores[batch_size:]
        rewards = np.repeat(scores[:, (np.newaxis)], gen_result.shape[1], 1)
        return rewards, _


class RewardCriterion(nn.Module):

    def __init__(self, opt):
        super(RewardCriterion, self).__init__()
        self.vocab_size = opt.vocab_size

    def forward(self, seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs,
        fgLogprobs, reward):
        seqLogprobs = seqLogprobs.view(-1)
        reward = reward.view(-1)
        fg_seq = fg_seq.squeeze()
        seq_mask = torch.cat((seq.new(seq.size(0), 1).fill_(1).byte(), seq.
            gt(0)[:, :-1]), 1).view(-1)
        seq_mask = Variable(seq_mask)
        seq_out = -torch.masked_select(seqLogprobs * reward, seq_mask)
        seq_out = torch.sum(seq_out) / torch.sum(seq_mask.data)
        bnLogprobs = bnLogprobs.view(-1)
        bn_mask = fg_seq.gt(0).view(-1)
        bn_mask = Variable(bn_mask)
        bn_out = -torch.masked_select(bnLogprobs * reward, bn_mask)
        bn_out = torch.sum(bn_out) / max(torch.sum(bn_mask.data), 1)
        fgLogprobs = fgLogprobs.view(-1)
        fg_out = -torch.masked_select(fgLogprobs * reward, bn_mask)
        fg_out = torch.sum(fg_out) / max(torch.sum(bn_mask.data), 1)
        return seq_out, bn_out, fg_out


class LMCriterion(nn.Module):

    def __init__(self, opt):
        super(LMCriterion, self).__init__()
        self.vocab_size = opt.vocab_size

    def forward(self, txt_input, vis_input, target):
        target_copy = target.clone()
        vis_mask = Variable(target.data > self.vocab_size).view(-1, 1)
        txt_mask = target.data.gt(0)
        txt_mask = torch.cat([txt_mask.new(txt_mask.size(0), 1).fill_(1),
            txt_mask[:, :-1]], 1)
        txt_mask[target.data > self.vocab_size] = 0
        vis_out = -torch.masked_select(vis_input, vis_mask)
        target.data[target.data > self.vocab_size] = 0
        target = target.view(-1, 1)
        txt_select = torch.gather(txt_input, 1, target)
        if isinstance(txt_input, Variable):
            txt_mask = Variable(txt_mask)
        txt_out = -torch.masked_select(txt_select, txt_mask.view(-1, 1))
        loss = (torch.sum(txt_out) + torch.sum(vis_out)).float() / (torch.
            sum(txt_mask.data) + torch.sum(vis_mask.data)).float()
        return loss


class BNCriterion(nn.Module):

    def __init__(self, opt):
        super(BNCriterion, self).__init__()

    def forward(self, input, target):
        target = target.view(-1, 1) - 1
        bn_mask = target.data.ne(-1)
        if isinstance(input, Variable):
            bn_mask = Variable(bn_mask)
        if torch.sum(bn_mask.data) > 0:
            new_target = target.data.clone()
            new_target[new_target < 0] = 0
            select = torch.gather(input.view(-1, 2), 1, Variable(new_target))
            out = -torch.masked_select(select, bn_mask)
            loss = torch.sum(out).float() / torch.sum(bn_mask.data).float()
        else:
            loss = Variable(input.data.new(1).zero_()).float()
        return loss


class FGCriterion(nn.Module):

    def __init__(self, opt):
        super(FGCriterion, self).__init__()

    def forward(self, input, target):
        target = target.view(-1, 1)
        input = input.view(-1, input.size(2))
        select = torch.gather(input, 1, target)
        attr_mask = target.data.gt(0)
        if isinstance(input, Variable):
            attr_mask = Variable(attr_mask)
        if torch.sum(attr_mask.data) > 0:
            out = -torch.masked_select(select, attr_mask)
            loss = torch.sum(out).float() / torch.sum(attr_mask.data).float()
        else:
            loss = Variable(input.data.new(1).zero_()).float()
        return loss


class vgg16(nn.Module):

    def __init__(self, opt, pretrained=True):
        super(vgg16, self).__init__()
        self.model_path = '%s/imagenet_weights/vgg16_caffe.pth' % opt.data_path
        self.pretrained = pretrained
        vgg = models.vgg16()
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values
            ())[:-1])
        self.fc = vgg.classifier
        self.pooling = nn.AdaptiveAvgPool2d((7, 7))
        if self.pretrained:
            None
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in
                vgg.state_dict()})
        self.cnn_net = nn.Sequential(*list(vgg.features._modules.values())[:-1]
            )

    def forward(self, img):
        conv_feat = self.cnn_net(img)
        pooled_conv_feat = self.pooling(conv_feat)
        pooled_conv_feat_flat = pooled_conv_feat.view(pooled_conv_feat.size
            (0), -1)
        fc_feat = self.fc(pooled_conv_feat_flat)
        return conv_feat, fc_feat


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height,
            self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.
            aligned_width, self.spatial_scale, grad_output, self.rois,
            grad_input)
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
            self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    """
    return precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        """ copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, df_mode='corpus', test=None, refs=None, n=4, sigma=6.0):
        """ singular instance """
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        if self.df_mode != 'corpus':
            self.document_frequency = pickle.load(open(os.path.join('data',
                df_mode + '.p'), 'r'))
        self.cook_append(test, refs)
        self.ref_len = None

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest
            ), 'refs/test mismatch! %d<>%d' % (len(self.crefs), len(self.ctest)
            )
        return len(self.crefs)

    def __iadd__(self, other):
        """add an instance (e.g., from another sentence)."""
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for ngram, count in ref
                .iteritems()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):

        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [(0.0) for _ in range(self.n)]
            for ngram, term_freq in cnts.iteritems():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            val = np.array([(0.0) for _ in range(self.n)])
            for n in range(self.n):
                for ngram, count in vec_hyp[n].iteritems():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= norm_hyp[n] * norm_ref[n]
                assert not math.isnan(val[n])
            return val
        if self.df_mode == 'corpus':
            self.ref_len = np.log(float(len(self.crefs)))
        elif self.df_mode == 'coco-val':
            self.ref_len = np.log(float(40504))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([(0.0) for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        if self.df_mode == 'corpus':
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            assert len(self.ctest) >= max(self.document_frequency.values())
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


class CiderD(nn.Module):
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, n=4, sigma=6.0, df='corpus'):
        super(CiderD, self).__init__()
        self._n = n
        self._sigma = sigma
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        self.cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            self.cider_scorer += hypo[0], ref
        score, scores = self.cider_scorer.compute_score()
        return score, scores

    def method(self):
        return 'CIDEr-D'


class CiderScorer(nn.Module):
    """CIDEr scorer.
    """

    def __init__(self, df_mode='corpus', test=None, refs=None, n=4, sigma=6.0):
        """ singular instance """
        super(CiderScorer, self).__init__()
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.ref_len = None
        if self.df_mode != 'corpus':
            pkl_file = pickle.load(open(os.path.join('data', df_mode + '.p'
                ), 'r'))
            self.ref_len = pkl_file['ref_len']
            self.document_frequency = pkl_file['document_frequency']
        self.cook_append(test, refs)

    def clear(self):
        self.crefs = []
        self.ctest = []

    def copy(self):
        """ copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest
            ), 'refs/test mismatch! %d<>%d' % (len(self.crefs), len(self.ctest)
            )
        return len(self.crefs)

    def __iadd__(self, other):
        """add an instance (e.g., from another sentence)."""
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for ngram, count in ref
                .iteritems()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):

        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [(0.0) for _ in range(self.n)]
            for ngram, term_freq in cnts.iteritems():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            val = np.array([(0.0) for _ in range(self.n)])
            for n in range(self.n):
                for ngram, count in vec_hyp[n].iteritems():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]
                        ) * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= norm_hyp[n] * norm_ref[n]
                assert not math.isnan(val[n])
                val[n] *= np.e ** (-delta ** 2 / (2 * self.sigma ** 2))
            return val
        if self.df_mode == 'corpus':
            self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([(0.0) for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        if self.df_mode == 'corpus':
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            assert len(self.ctest) >= max(self.document_frequency.values())
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jiasenlu_NeuralBabyTalk(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BNCriterion(*[], **{'opt': 4}), [torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FGCriterion(*[], **{'opt': 4}), [torch.zeros([4, 4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

