import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
dataloaderraw = _module
eval = _module
eval_ensemble = _module
eval_multi = _module
eval_utils = _module
misc = _module
config = _module
div_utils = _module
loss_wrapper = _module
resnet = _module
resnet_utils = _module
rewards = _module
utils = _module
AttEnsemble = _module
AttModel = _module
BertCapModel = _module
CaptionModel = _module
FCModel = _module
M2Transformer = _module
ShowTellModel = _module
TransformerModel = _module
models = _module
opts = _module
build_bpe_subword_nmt = _module
dump_to_h5df = _module
dump_to_lmdb = _module
make_bu_data = _module
prepro_feats = _module
prepro_labels = _module
prepro_ngrams = _module
prepro_reference_json = _module
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


import torch.nn as nn


import numpy as np


import random


import string


import torch.nn.functional as F


import collections


import torch.optim as optim


from torch.autograd import *


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import copy


import math


from torch.utils.tensorboard import SummaryWriter


from collections import defaultdict


CiderD_scorer = None


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


Bleu_scorer = None


def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts)
    assert greedy_res.shape[0] == batch_size
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))
            ]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({(i + gen_result_size): gts[i] for i in range(batch_size)})
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = (opt.cider_reward_weight * cider_scores + opt.
        bleu_reward_weight * bleu_scores)
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img
        ) - scores[-batch_size:][:, (np.newaxis)]
    scores = scores.reshape(gen_result_size)
    rewards = np.repeat(scores[:, (np.newaxis)], gen_result.shape[1], 1)
    return rewards


class LossWrapper(torch.nn.Module):

    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.struc_crit = utils.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts,
        gt_indices, sc_flag, struc_flag):
        opt = self.opt
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels,
                    att_masks), labels[(...), 1:], masks[(...), 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats,
                    att_feats, att_masks, opt={'sample_method': opt.
                    train_sample_method, 'beam_size': opt.train_beam_size,
                    'output_logsoftmax': opt.struc_use_logsoftmax or opt.
                    structure_loss_type == 'softmax_margin' or not 'margin' in
                    opt.structure_loss_type, 'sample_n': opt.train_sample_n
                    }, mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                    'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1 - opt.structure_loss_weight
                ) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels,
                att_masks), labels[(...), 1:], masks[(...), 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample', opt={'sample_method': opt.
                    sc_sample_method, 'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats,
                att_masks, opt={'sample_method': opt.train_sample_method,
                'beam_size': opt.train_beam_size, 'sample_n': opt.
                train_sample_n}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result,
                self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:, (0)].mean()
        out['loss'] = loss
        return out


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
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(
            1, 2, 0)
        return fc, att


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1
            ).reshape(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


Cider_scorer = None


def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data_gts)
    res = []
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))
    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i * seq_per_img:(i + 1) *
            seq_per_img]])

        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / np.sqrt(eigvals).sum()
                ) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0] / 10)))
    scores = np.array(scores)
    return scores


def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data_gts)
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))
            ]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = (opt.cider_reward_weight * cider_scores + opt.
        bleu_reward_weight * bleu_scores)
    return scores


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """

    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}
        batch_size = input.size(0)
        seq_per_img = batch_size // len(data_gts)
        assert seq_per_img == self.opt.train_sample_n, seq_per_img
        mask = (seq > 0).float()
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1
            )
        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores
        if self.opt.entropy_reward_weight > 0:
            entropy = -(F.softmax(input, dim=2) * F.log_softmax(input, dim=2)
                ).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            None
            scores = scores + self.opt.entropy_reward_weight * entropy.view(
                -1, seq_per_img)
        costs = -scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin':
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        if self.loss_type == 'seqnll':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'risk':
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)
            output = (F.softmax(input.exp()) * costs).sum(1).mean()
        elif self.loss_type == 'max_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0
                ] / 2
            output = output.mean()
        elif self.loss_type == 'multi_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()
        elif self.loss_type == 'softmax_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'real_softmax_margin':
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.
                shape[1] - 1)
            scores = scores - baseline
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = -input * mask * scores.view(-1, 1)
            output = torch.sum(output) / torch.sum(mask)
        out['loss'] = output
        return out


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1)
        self.size = input.size(1)
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return (self.criterion(input, true_dist).sum(1) * mask).sum(
            ) / mask.sum()


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
        self.w2h = nn.Linear(self.input_encoding_size, (4 + (use_maxout == 
            True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4 + (use_maxout == True)) *
            self.rnn_size)
        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout ==
            True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4 + (use_maxout ==
            True)) * self.rnn_size) for _ in range(self.num_layers)])
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
            sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size
                )
            if not self.use_maxout:
                in_transform = torch.tanh(all_input_sums.narrow(1, 3 * self
                    .rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 
                    2 * self.rnn_size)
                in_transform = torch.max(in_transform.narrow(1, 0, self.
                    rnn_size), in_transform.narrow(1, self.rnn_size, self.
                    rnn_size))
            next_c = forget_gate * prev_c + in_gate * in_transform
            tanh_nex_c = torch.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers - 1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h + self.r_h2h(prev_h)
                fake_region = torch.sigmoid(n5) * tanh_nex_c
            cs.append(next_c)
            hs.append(next_h)
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)
        state = torch.cat([_.unsqueeze(0) for _ in hs], 0), torch.cat([_.
            unsqueeze(0) for _ in cs], 0)
        return top_h, fake_region, state


class AdaAtt_attention(nn.Module):

    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.fr_linear = nn.Sequential(nn.Linear(self.rnn_size, self.
            input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.ho_linear = nn.Sequential(nn.Linear(self.rnn_size, self.
            input_encoding_size), nn.Tanh(), nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed,
        att_masks=None):
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)
        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)
        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0),
            att_size + 1, h_out_embed.size(1))
        img_all = torch.cat([fake_region.view(-1, 1, self.
            input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.
            input_encoding_size), conv_feat_embed], 1)
        hA = torch.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA, self.drop_prob_lm, self.training)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)
        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:, :1], att_masks], 1)
            PI = PI / PI.sum(1, keepdim=True)
        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)
        atten_out = visAttdim + h_out_linear
        h = torch.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class AdaAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats,
            att_masks)
        return atten_out, state


class UpDownCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size *
            2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0])
            )
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1],
            state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])
        return output, state


class StackAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)
        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0
            :1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0]
            [1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats,
            p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1, att_res_2], 1), [state[0]
            [2:3], state[1][2:3]])
        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]


class DenseAttCore(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)
        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size * 2, opt.
            rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm))
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size * 3, opt.
            rnn_size), nn.ReLU(), nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0
            :1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0]
            [1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats,
            p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0,
            h_1], 1)), att_res_2], 1), [state[0][2:3], state[1][2:3]])
        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0
            ) for _ in zip(state_0, state_1, state_2)]


class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)
        return att_res


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

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats,
            att_masks)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self
            .rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class Att2all2Core(nn.Module):

    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state,
        att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats,
            att_masks)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(
            att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self
            .rnn_size)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(inputs_embeds=src, attention_mask=src_mask)[0]

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(input_ids=tgt, attention_mask=tgt_mask,
            encoder_hidden_states=memory, encoder_attention_mask=src_mask)[0]


class CaptionModel(nn.Module):

    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobs, t, divm,
            diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]
            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, (
                        local_time)]
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, (
                            prev_labels)].unsqueeze(-1), change.new_ones(
                            batch_size, 1))
                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change
                        ) * diversity_lambda
            return logprobs, unaug_logprobs

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq,
            beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs
            ys, ix = torch.sort(candidate_logprobs.reshape(
                candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size
            selected_ix = ix % vocab_size
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix)
                .unsqueeze(-1) * logprobs.shape[1]).reshape(-1)
            if t > 0:
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(
                    beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[
                    state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).
                    expand_as(beam_seq))
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.
                    unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))
            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix
                ) + logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size,
                -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size
                ).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size))
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([beam_seq_logprobs, beam_logprobs
                .reshape(batch_size, -1, 1, vocab_size)], 2)
            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                new_state[_ix] = state[_ix][:, (state_ix)]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size
        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for
            _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, 
            self.vocab_size + 1).to(device) for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device
            ) for _ in range(group_size)]
        done_beams_table = [[[] for __ in range(group_size)] for _ in range
            (batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(
            group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        args = list(args)
        args = utils.split_tensors(group_size, args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in
                range(len(args))] for k in range(group_size)]
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(
                group_size)]
        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobs = logprobs_table[divm]
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, (t -
                            divm - 1)].reshape(-1, 1).to(device), float('-inf')
                            )
                    if remove_bad_endings and t - divm > 0:
                        logprobs[torch.from_numpy(np.isin(beam_seq_table[
                            divm][:, :, (t - divm - 1)].cpu().numpy(), self
                            .bad_endings_ix)).reshape(-1), 0] = float('-inf')
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[
                        str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, (logprobs.size(1) - 1)] = logprobs[:, (
                            logprobs.size(1) - 1)] - 1000
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table,
                        logprobs, t, divm, diversity_lambda, bdash)
                    beam_seq_table[divm], beam_seq_logprobs_table[divm
                        ], beam_logprobs_sum_table[divm], state_table[divm
                        ] = beam_step(logprobs, unaug_logprobs, bdash, t -
                        divm, beam_seq_table[divm], beam_seq_logprobs_table
                        [divm], beam_logprobs_sum_table[divm], state_table[
                        divm])
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][(b), :, (t - divm)] == 0
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == self.seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {'seq': beam_seq_table[divm][b,
                                    vix].clone(), 'logps':
                                    beam_seq_logprobs_table[divm][b, vix].
                                    clone(), 'unaug_p':
                                    beam_seq_logprobs_table[divm][b, vix].
                                    sum().item(), 'p':
                                    beam_logprobs_sum_table[divm][b, vix].
                                    item()}
                                final_beam['p'] = length_penalty(t - divm +
                                    1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000
                    it = beam_seq_table[divm][:, :, (t - divm)].reshape(-1)
                    logprobs_table[divm], state_table[divm
                        ] = self.get_logprobs_state(it, *(args[divm] + [
                        state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[
                        divm] / temperature, dim=-1)
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -
            x['p'])[:bdash] for i in range(group_size)] for b in range(
            batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def old_beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobsf, t, divm,
            diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]
                            ] = logprobsf[sub_beam][prev_decisions[prev_labels]
                            ] - diversity_lambda
            return unaug_logprobsf

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq,
            beam_seq_logprobs, beam_logprobs_sum, state):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p':
                        candidate_logprob, 'r': unaug_logprobsf[q]})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, (vix)] = beam_seq_prev[:, (v['q'])]
                    beam_seq_logprobs[:t, (vix)] = beam_seq_logprobs_prev[:,
                        (v['q'])]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, (vix)] = state[state_ix][:, (v['q'])
                        ]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            return (beam_seq, beam_seq_logprobs, beam_logprobs_sum, state,
                candidates)
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for
            _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash,
            self.vocab_size + 1).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(
            group_size)]
        done_beams_table = [[] for _ in range(group_size)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[(_.chunk(group_size) if _ is not None else [None] *
                group_size) for _ in args_] for args_ in args]
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in
                range(len(args))] for k in range(group_size)]
        else:
            args = [(_.chunk(group_size) if _ is not None else [None] *
                group_size) for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(
                group_size)]
        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobsf = logprobs_table[divm].float()
                    if decoding_constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm -
                            1].unsqueeze(1), float('-inf'))
                    if remove_bad_endings and t - divm > 0:
                        logprobsf[torch.from_numpy(np.isin(beam_seq_table[
                            divm][t - divm - 1].cpu().numpy(), self.
                            bad_endings_ix)), 0] = float('-inf')
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[
                        str(logprobsf.size(1) - 1)] == 'UNK':
                        logprobsf[:, (logprobsf.size(1) - 1)] = logprobsf[:,
                            (logprobsf.size(1) - 1)] - 1000
                    unaug_logprobsf = add_diversity(beam_seq_table,
                        logprobsf, t, divm, diversity_lambda, bdash)
                    beam_seq_table[divm], beam_seq_logprobs_table[divm
                        ], beam_logprobs_sum_table[divm], state_table[divm
                        ], candidates_divm = beam_step(logprobsf,
                        unaug_logprobsf, bdash, t - divm, beam_seq_table[
                        divm], beam_seq_logprobs_table[divm],
                        beam_logprobs_sum_table[divm], state_table[divm])
                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix
                            ] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {'seq': beam_seq_table[divm][:, (
                                vix)].clone(), 'logps':
                                beam_seq_logprobs_table[divm][:, (vix)].
                                clone(), 'unaug_p': beam_seq_logprobs_table
                                [divm][:, (vix)].sum().item(), 'p':
                                beam_logprobs_sum_table[divm][vix].item()}
                            final_beam['p'] = length_penalty(t - divm + 1,
                                final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000
                    it = beam_seq_table[divm][t - divm]
                    logprobs_table[divm], state_table[divm
                        ] = self.get_logprobs_state(it, *(args[divm] + [
                        state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[
                        divm] / temperature, dim=-1)
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x[
            'p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel':

            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape)
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith('top'):
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs,
                        descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:,
                        :-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1,
                        keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()
                ).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        return it, sampleLogprobs

    def decode_sequence(self, seq):
        return utils.decode_sequence(self.vocab, seq)


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
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        in_transform = torch.max(all_input_sums.narrow(1, 3 * self.rnn_size,
            self.rnn_size), all_input_sums.narrow(1, 4 * self.rnn_size,
            self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)
        output = self.dropout(next_h)
        state = next_h.unsqueeze(0), next_c.unsqueeze(0)
        return output, state


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


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


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


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


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).
            transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self
            .dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k
            )
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.
            log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ruotianluo_self_critical_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Embeddings(*[], **{'d_model': 4, 'vocab': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(Generator(*[], **{'d_model': 4, 'vocab': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(LayerNorm(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MultiHeadedAttention(*[], **{'h': 4, 'd_model': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(PositionalEncoding(*[], **{'d_model': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(PositionwiseFeedForward(*[], **{'d_model': 4, 'd_ff': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(SublayerConnection(*[], **{'size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), ReLU()], {})

