import sys
_module = sys.modules[__name__]
del sys
OpenAttack = _module
attack_assist = _module
filter_words = _module
chinese = _module
english = _module
goal = _module
base = _module
classifier_goal = _module
substitute = _module
char = _module
chinese_fyh_char = _module
chinese_sim_char = _module
english_dces = _module
english_eces = _module
word = _module
chinese_cilin = _module
chinese_hownet = _module
chinese_word2vec = _module
chinese_wordnet = _module
embed_based = _module
english_counterfit = _module
english_glove = _module
english_hownet = _module
english_word2vec = _module
english_wordnet = _module
word_embedding = _module
attack_eval = _module
utils = _module
attackers = _module
bae = _module
bert_attack = _module
classification = _module
deepwordbug = _module
fd = _module
gan = _module
genetic = _module
geometry = _module
hotflip = _module
pso = _module
pwws = _module
scpn = _module
models = _module
subword = _module
textbugger = _module
textfooler = _module
uat = _module
viper = _module
data = _module
cilin_dict = _module
counter_fit = _module
dces = _module
fyh_dict = _module
gan = _module
glove = _module
hownet = _module
hownet_substitute_dict = _module
nltk_perceptron_pos_tagger = _module
nltk_senttokenizer = _module
nltk_wordnet = _module
nltk_wordnet_delemma = _module
sentence_transformer = _module
sgan = _module
sim_dict = _module
stanford_ner = _module
stanford_parser = _module
test = _module
translation_models = _module
universal_sentence_encoder = _module
victim_albert_ag = _module
victim_albert_imdb = _module
victim_albert_sst = _module
victim_bert = _module
victim_bert_amazon_zh = _module
victim_roberta_ag = _module
victim_roberta_imdb = _module
victim_roberta_sst = _module
victim_xlnet_ag = _module
victim_xlnet_imdb = _module
victim_xlnet_sst = _module
word2vec = _module
data_manager = _module
exception = _module
exceptions = _module
victim = _module
metric = _module
algorithms = _module
bleu = _module
gptlm = _module
jaccard_char = _module
jaccard_word = _module
language_tool = _module
levenshtein = _module
modification = _module
sentence_sim = _module
usencoder = _module
selectors = _module
edit_distance = _module
fluency = _module
grammar = _module
modify = _module
semantic = _module
tags = _module
text_process = _module
constituency_parser = _module
lemmatizer = _module
wordnet_lemmatizer = _module
tokenizer = _module
jieba_tokenizer = _module
punct_tokenizer = _module
transformers_tokenizer = _module
auto_lang = _module
transform_label = _module
transformers_hook = _module
visualizer = _module
zip_downloader = _module
version = _module
classifiers = _module
methods = _module
transformers = _module
context = _module
method = _module
demo = _module
demo_chinese = _module
demo_deo = _module
conf = _module
adversarial_training = _module
custom_attacker = _module
custom_dataset = _module
custom_metrics = _module
custom_victim = _module
multiprocess_eval = _module
nli_attack = _module
transformer = _module
workflow = _module
setup = _module
attackers_chinese = _module
test_chinese_default = _module
test_chinese_multi_process = _module
test_default = _module
test_multi_process = _module
run_test = _module
test_default_text_processor = _module
test_meta_classifier = _module

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


from typing import Union


import torch


from typing import Dict


from typing import Optional


from typing import List


import copy


import random


import numpy as np


from copy import deepcopy


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


import time


import string


from collections import Counter


from torch.nn import CosineSimilarity


from torch.nn.utils.rnn import pad_packed_sequence as unpack


from torch.nn.utils.rnn import pack_padded_sequence as pack


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


class DeepFool(nn.Module):

    def __init__(self, config, num_classes, max_iters, overshoot=0.02):
        super(DeepFool, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.loops_needed = None
        self.max_iters = max_iters
        self.overshoot = overshoot
        self.loops = 0

    def forward(self, vecs, net_, target=None):
        """

        :param vecs: [batch_size, vec_size]
        :param net_: FFNN in our case
        :param target:
        :return:
        """
        net = deepcopy(net_.classifier)
        sent_vecs = deepcopy(vecs.data)
        input_shape = sent_vecs.size()
        f_vecs = net.forward(sent_vecs).data
        I = torch.argsort(f_vecs, dim=1, descending=True)
        I = I[:, 0:self.num_classes]
        label = I[:, 0]
        if target is not None:
            I = target.unsqueeze(1)
            if self.config['dataset'] == 'imdb':
                num_classes = 2
            elif self.config['dataset'] == 'agnews':
                num_classes = 4
            else:
                None
        else:
            num_classes = I.size(1)
        pert_vecs = deepcopy(sent_vecs)
        r_tot = torch.zeros(input_shape)
        check_fool = deepcopy(sent_vecs)
        k_i = label
        loop_i = 0
        finish_mask = torch.zeros((input_shape[0], 1), dtype=torch.float)
        finished = torch.ones_like(finish_mask)
        self.loops_needed = torch.zeros((input_shape[0],))
        if torch.cuda.is_available():
            r_tot = r_tot
            finish_mask = finish_mask
            finished = finished
            self.loops_needed = self.loops_needed
        while torch.sum(finish_mask >= finished) != input_shape[0] and loop_i < self.max_iters:
            x = pert_vecs.requires_grad_(True)
            fs = net.forward(x)
            pert = torch.ones(input_shape[0]) * np.inf
            w = torch.zeros(input_shape)
            if torch.cuda.is_available():
                pert = pert
                w = w
            logits_label_sum = torch.gather(fs, dim=1, index=label.unsqueeze(1)).sum()
            logits_label_sum.backward(retain_graph=True)
            grad_orig = deepcopy(x.grad.data)
            for k in range(1, num_classes):
                if target is not None:
                    k = k - 1
                    if k > 0:
                        break
                zero_gradients(x)
                logits_class_sum = torch.gather(fs, dim=1, index=I[:, k].unsqueeze(1)).sum()
                logits_class_sum.backward(retain_graph=True)
                cur_grad = deepcopy(x.grad.data)
                w_k = cur_grad - grad_orig
                f_k = torch.gather(fs, dim=1, index=I[:, k].unsqueeze(1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))
                f_k = f_k.squeeze(-1)
                pert_k = torch.div(torch.abs(f_k), self.norm_dim(w_k))
                valid_pert_mask = pert_k < pert
                new_pert = pert_k + 0.0
                new_w = w_k + 0.0
                valid_pert_mask = valid_pert_mask.bool()
                pert = torch.where(valid_pert_mask, new_pert, pert)
                valid_w_mask = torch.reshape(valid_pert_mask, shape=(input_shape[0], 1)).float()
                valid_w_mask = valid_w_mask.bool()
                w = torch.where(valid_w_mask, new_w, w)
            r_i = torch.mul(torch.clamp(pert, min=0.0001).reshape(-1, 1), w)
            r_i = torch.div(r_i, self.norm_dim(w).reshape((-1, 1)))
            r_tot_new = r_tot + r_i
            cur_update_mask = (finish_mask < 1.0).byte()
            if torch.cuda.is_available():
                cur_update_mask = cur_update_mask
            cur_update_mask = cur_update_mask.bool()
            r_tot = torch.where(cur_update_mask, r_tot_new, r_tot)
            pert_vecs = sent_vecs + r_tot
            check_fool = sent_vecs + (1.0 + self.overshoot) * r_tot
            k_i = torch.argmax(net.forward(check_fool.requires_grad_(True)), dim=-1).data
            if target is None:
                finish_mask += ((k_i != label) * 1.0).reshape((-1, 1)).float()
            else:
                finish_mask += ((k_i == target) * 1.0).reshape((-1, 1)).float()
            loop_i += 1
            self.loops += 1
            self.loops_needed[cur_update_mask.squeeze()] = loop_i
            r_tot.detach_()
            check_fool.detach_()
            r_i.detach_()
            pert_vecs.detach_()
        x = pert_vecs.requires_grad_(True)
        fs = net.forward(x)
        torch.sum(torch.gather(fs, dim=1, index=k_i.unsqueeze(1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))).backward(retain_graph=True)
        grad = deepcopy(x.grad.data)
        grad = torch.div(grad, self.norm_dim(grad).unsqueeze(1))
        label = deepcopy(label.data)
        if target is not None:
            pert_vecs = deepcopy(pert_vecs.data)
            return grad, pert_vecs, label
        else:
            check_fool_vecs = deepcopy(check_fool.data)
            return grad, check_fool_vecs, label

    @staticmethod
    def norm_dim(w):
        norms = []
        for idx in range(w.size(0)):
            norms.append(w[idx].norm())
        norms = torch.stack(tuple(norms), dim=0)
        return norms


class ParseNet(nn.Module):

    def __init__(self, d_nt, d_hid, len_voc):
        super(ParseNet, self).__init__()
        self.d_nt = d_nt
        self.d_hid = d_hid
        self.len_voc = len_voc
        self.trans_embs = nn.Embedding(len_voc, d_nt)
        self.encoder = nn.LSTM(d_nt, d_hid, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(d_nt + d_hid, d_hid, num_layers=1, batch_first=True)
        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.out_nonlin = nn.LogSoftmax(dim=1)
        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_nt + d_hid, 1))

    def compute_mask(self, lengths):
        device = lengths.device
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len, device=device).unsqueeze(0).expand(lengths.size(0), max_len)
        mask = lengths.unsqueeze(1).expand_as(range_row)
        mask = (range_row < mask).float()
        return mask

    def masked_softmax(self, vector, mask):
        result = nn.functional.softmax(vector, dim=1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
        return result

    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous[0].mm(self.att_W)
        scores = b_hn.unsqueeze(1) * enc_hids
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):
        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous[0].mm(self.att_parse_W)
        scores = b_hn.unsqueeze(1) * trans_embs
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    def encode_batch(self, inputs, lengths):
        device = inputs.device
        bsz, max_len = inputs.size()
        in_embs = self.trans_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        e_hid_init = torch.zeros(1, bsz, self.d_hid, device=device)
        e_cell_init = torch.zeros(1, bsz, self.d_hid, device=device)
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0]
        return all_hids[_indices], enc_last_hid.squeeze(0)[_indices]

    def decode_step(self, idx, prev_words, prev_hid, prev_cell, enc_hids, trans_embs, in_sent_lens, trans_lens, bsz, max_len):
        device = self.trans_embs.parameters().__next__().device
        if idx == 0:
            word_input = torch.zeros(bsz, 1, self.d_nt, device=device)
        else:
            word_input = self.trans_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_nt)
        trans_weights = self.compute_transformation_attention(prev_hid, trans_embs, trans_lens)
        trans_ctx = torch.sum(trans_weights.unsqueeze(2) * trans_embs, dim=1)
        decoder_input = torch.cat([word_input, trans_ctx.unsqueeze(1)], dim=2)
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))
        attn_weights = self.compute_decoder_attention(hn, enc_hids, in_sent_lens)
        attn_ctx = torch.sum(attn_weights.unsqueeze(2) * enc_hids, dim=1)
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)
        p_copy += attn_ctx.mm(self.copy_att_v)
        p_copy += hn.squeeze(0).mm(self.copy_hid_v)
        p_copy = torch.sigmoid(p_copy).squeeze(1)
        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self):
        raise NotImplemented

    def batch_beam_search(self, inputs, out_trimmed, in_trans_lens, out_trimmed_lens, eos_idx, beam_size=5, max_steps=250):
        device = inputs.device
        bsz, max_len = inputs.size()
        inputs = inputs[:, :in_trans_lens[0]]
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_lens)
        trim_hids, trim_last_hid = self.encode_batch(out_trimmed, out_trimmed_lens)
        hn = enc_last_hid.unsqueeze(0)
        cn = torch.zeros(1, 1, self.d_hid, device=device)
        beam_dict = {}
        for b_idx in range(trim_hids.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0
        while True:
            prev_words = []
            prev_hs = []
            prev_cs = []
            for b_idx in beam_dict:
                beams = beam_dict[b_idx]
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None
                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)
            hs = torch.cat(prev_hs, dim=1)
            cs = torch.cat(prev_cs, dim=1)
            num_examples = hs.size(1)
            if prev_words is not None:
                prev_words = torch.LongTensor(prev_words)
            if num_examples != trim_hids.size(0):
                d1, d2, d3 = trim_hids.size()
                rep_factor = num_examples // d1
                curr_out = trim_hids.unsqueeze(1).expand(d1, rep_factor, d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens = out_trimmed_lens.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)
            else:
                curr_out = trim_hids
                curr_out_lens = out_trimmed_lens
            _, in_len, hid_d = enc_hids.size()
            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)
            curr_enc_lens = in_trans_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_trans_lens[0])
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words, hs, cs, curr_enc_hids, curr_out, curr_enc_lens, curr_out_lens, num_examples, max_len)
            vocab_scores = torch.zeros(num_examples, self.len_voc, device=device)
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()
            pred_inp = torch.cat([hn.squeeze(0), attn_ctx], dim=1)
            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy.unsqueeze(1) * vocab_scores + (1 - p_copy.unsqueeze(1)) * preds
            for b_idx in beam_dict:
                beam_candidates = []
                if num_examples == len(beam_dict):
                    ex_hn = hn[:, b_idx, :].unsqueeze(0)
                    ex_cn = cn[:, b_idx, :].unsqueeze(0)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    for z in range(beam_size):
                        word_idx = top_indices[z].item()
                        beam_candidates.append((preds[word_idx].item(), ex_hn, ex_cn, [word_idx]))
                    beam_dict[b_idx] = beam_candidates
                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:, start:end, :]
                    ex_cn = cn[:, start:end, :]
                    ex_preds = final_preds[start:end]
                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)
                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:, o_idx, :].unsqueeze(0)
                        curr_cn = ex_cn[:, o_idx, :].unsqueeze(0)
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].item()
                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().item()), curr_hn, curr_cn, seq + [word_idx]))
                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]
            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict


class SCPN(nn.Module):

    def __init__(self, d_word, d_hid, d_nt, d_trans, len_voc, len_trans_voc, use_input_parse):
        super(SCPN, self).__init__()
        self.d_word = d_word
        self.d_hid = d_hid
        self.d_trans = d_trans
        self.d_nt = d_nt + 1
        self.len_voc = len_voc
        self.len_trans_voc = len_trans_voc
        self.use_input_parse = use_input_parse
        self.word_embs = nn.Embedding(len_voc, d_word)
        self.trans_embs = nn.Embedding(len_trans_voc, d_nt)
        if use_input_parse:
            self.encoder = nn.LSTM(d_word + d_trans, d_hid, num_layers=1, bidirectional=True, batch_first=True)
        else:
            self.encoder = nn.LSTM(d_word, d_hid, num_layers=1, bidirectional=True, batch_first=True)
        self.encoder_proj = nn.Linear(d_hid * 2, d_hid)
        self.decoder = nn.LSTM(d_word + d_hid, d_hid, num_layers=2, batch_first=True)
        self.trans_encoder = nn.LSTM(d_nt, d_trans, num_layers=1, batch_first=True)
        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.att_nonlin = nn.Softmax(dim=1)
        self.out_nonlin = nn.LogSoftmax(dim=1)
        self.att_parse_proj = nn.Linear(d_trans, d_hid)
        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid))
        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1))
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_word + d_hid, 1))

    def compute_mask(self, lengths):
        device = lengths.device
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len, device=device).unsqueeze(0).expand(lengths.size(0), max_len)
        mask = lengths.unsqueeze(1).expand_as(range_row)
        mask = (range_row < mask).float()
        return mask

    def masked_softmax(self, vector, mask):
        result = torch.nn.functional.softmax(vector, dim=1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
        return result

    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous.mm(self.att_W)
        scores = b_hn.unsqueeze(1) * enc_hids
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):
        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous.mm(self.att_parse_W)
        scores = b_hn.unsqueeze(1) * trans_embs
        scores = torch.sum(scores, dim=2)
        scores = self.masked_softmax(scores, mask)
        return scores

    def encode_batch(self, inputs, trans, lengths):
        device = inputs.device
        bsz, max_len = inputs.size()
        in_embs = self.word_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        if self.use_input_parse:
            in_embs = torch.cat([in_embs, trans.unsqueeze(1).expand(bsz, max_len, self.d_trans)], dim=2)
        e_hid_init = torch.zeros(2, bsz, self.d_hid, device=device)
        e_cell_init = torch.zeros(2, bsz, self.d_hid, device=device)
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]
        all_hids = self.encoder_proj(all_hids.view(-1, self.d_hid * 2)).view(bsz, max_len, self.d_hid)
        enc_last_hid = torch.cat([enc_last_hid[0], enc_last_hid[1]], dim=1)
        enc_last_hid = self.encoder_proj(enc_last_hid)[_indices]
        return all_hids, enc_last_hid

    def encode_transformations(self, trans, lengths, return_last=True):
        device = trans.device
        bsz, _ = trans.size()
        lens, indices = torch.sort(lengths, 0, True)
        in_embs = self.trans_embs(trans)
        t_hid_init = torch.zeros(1, bsz, self.d_trans, device=device)
        t_cell_init = torch.zeros(1, bsz, self.d_trans, device=device)
        all_hids, (enc_last_hid, _) = self.trans_encoder(pack(in_embs[indices], lens.tolist(), batch_first=True), (t_hid_init, t_cell_init))
        _, _indices = torch.sort(indices, 0)
        if return_last:
            return enc_last_hid.squeeze(0)[_indices]
        else:
            all_hids = unpack(all_hids, batch_first=True)[0]
            return all_hids[_indices]

    def decode_step(self, idx, prev_words, prev_hid, prev_cell, enc_hids, trans_embs, in_sent_lens, trans_lens, bsz, max_len):
        device = self.word_embs.parameters().__next__().device
        if idx == 0:
            word_input = torch.zeros(bsz, 1, self.d_word, device=device)
        else:
            word_input = self.word_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_word)
        trans_weights = self.compute_transformation_attention(prev_hid[1], trans_embs, trans_lens)
        trans_ctx = torch.sum(trans_weights.unsqueeze(2) * trans_embs, dim=1)
        decoder_input = torch.cat([word_input, trans_ctx.unsqueeze(1)], dim=2)
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))
        attn_weights = self.compute_decoder_attention(hn[1], enc_hids, in_sent_lens)
        attn_ctx = torch.sum(attn_weights.unsqueeze(2) * enc_hids, dim=1)
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)
        p_copy += attn_ctx.mm(self.copy_att_v)
        p_copy += hn[1].mm(self.copy_hid_v)
        p_copy = torch.sigmoid(p_copy).squeeze(1)
        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self):
        raise NotImplemented

    def batch_beam_search(self, inputs, out_trans, in_sent_lens, out_trans_lens, eos_idx, beam_size=5, max_steps=70):
        device = inputs.device
        bsz, max_len = inputs.size()
        inputs = inputs[:, :in_sent_lens[0]]
        out_trans_hids = self.encode_transformations(out_trans, out_trans_lens, return_last=False)
        out_trans_hids = self.att_parse_proj(out_trans_hids)
        enc_hids, enc_last_hid = self.encode_batch(inputs, None, in_sent_lens)
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = torch.zeros(2, 1, self.d_hid, device=device)
        beam_dict = {}
        for b_idx in range(out_trans.size(0)):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0
        while True:
            prev_words = []
            prev_hs = []
            prev_cs = []
            for b_idx in beam_dict:
                beams = beam_dict[b_idx]
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b
                    if len(seq) > 0:
                        prev_words.append(seq[-1])
                    else:
                        prev_words = None
                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)
            hs = torch.cat(prev_hs, dim=1)
            cs = torch.cat(prev_cs, dim=1)
            num_examples = hs.size(1)
            if prev_words is not None:
                prev_words = torch.LongTensor(prev_words)
            if num_examples != out_trans_hids.size(0):
                d1, d2, d3 = out_trans_hids.size()
                rep_factor = num_examples // d1
                curr_out = out_trans_hids.unsqueeze(1).expand(d1, rep_factor, d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens = out_trans_lens.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)
            else:
                curr_out = out_trans_hids
                curr_out_lens = out_trans_lens
            _, in_len, hid_d = enc_hids.size()
            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)
            curr_enc_lens = in_sent_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_sent_lens[0])
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words, hs, cs, curr_enc_hids, curr_out, curr_enc_lens, curr_out_lens, num_examples, max_len)
            vocab_scores = torch.zeros(num_examples, self.len_voc, device=device)
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()
            pred_inp = torch.cat([hn[1], attn_ctx], dim=1)
            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy.unsqueeze(1) * vocab_scores + (1 - p_copy.unsqueeze(1)) * preds
            for b_idx in beam_dict:
                beam_candidates = []
                if num_examples == len(beam_dict):
                    ex_hn = hn[:, b_idx, :].unsqueeze(1)
                    ex_cn = cn[:, b_idx, :].unsqueeze(1)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    for z in range(beam_size):
                        word_idx = top_indices[z].item()
                        beam_candidates.append((preds[word_idx].item(), ex_hn, ex_cn, [word_idx]))
                        beam_dict[b_idx] = beam_candidates
                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:, start:end, :]
                    ex_cn = cn[:, start:end, :]
                    ex_preds = final_preds[start:end]
                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)
                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:, o_idx, :]
                        curr_cn = ex_cn[:, o_idx, :]
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].item()
                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().item()), curr_hn.unsqueeze(1), curr_cn.unsqueeze(1), seq + [word_idx]))
                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]
            nsteps += 1
            if nsteps > max_steps:
                break
        return beam_dict

