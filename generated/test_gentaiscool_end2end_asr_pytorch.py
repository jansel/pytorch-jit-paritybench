import sys
_module = sys.modules[__name__]
del sys
aishell = _module
helper = _module
librispeech = _module
utils = _module
models = _module
asr = _module
transformer = _module
common_layers = _module
test = _module
train = _module
trainer = _module
trainer = _module
audio = _module
awd_lstm_utils = _module
constant = _module
data_loader = _module
functions = _module
lm_data_loader = _module
lm_functions = _module
logger = _module
lstm_utils = _module
metrics = _module
optimizer = _module

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


import torch.nn.functional as F


import torch.nn.init as I


from torch.autograd import Variable


import numpy as np


import math


import logging


class Transformer(nn.Module):
    """
    Transformer class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, decoder, feat_extractor='vgg_cnn'):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id2label = decoder.id2label
        self.feat_extractor = feat_extractor
        if feat_extractor == 'emb_cnn':
            self.conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(41, 11),
                stride=(2, 2), padding=(0, 10)), nn.BatchNorm2d(32), nn.
                Hardtanh(0, 20, inplace=True), nn.Conv2d(32, 32,
                kernel_size=(21, 11), stride=(2, 1)), nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True))
        elif feat_extractor == 'vgg_cnn':
            self.conv = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding
                =1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d(64, 128, 3,
                stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target, verbose=False
        ):
        """
        args:
            padded_input: B x 1 (channel for spectrogram=1) x (freq) x T
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            pred: B x T x vocab
            gold: B x T
        """
        if (self.feat_extractor == 'emb_cnn' or self.feat_extractor ==
            'vgg_cnn'):
            padded_input = self.conv(padded_input)
        sizes = padded_input.size()
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2],
            sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
            input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)
        hyp_seq = hyp_best_ids.squeeze(2)
        gold_seq = gold
        return pred, gold, hyp_seq, gold_seq

    def evaluate(self, padded_input, input_lengths, padded_target,
        beam_search=False, beam_width=0, beam_nbest=0, lm=None,
        lm_rescoring=False, lm_weight=0.1, c_weight=1, verbose=False):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            batch_ids_nbest_hyps: list of nbest id
            batch_strs_nbest_hyps: list of nbest str
            batch_strs_gold: list of gold str
        """
        if (self.feat_extractor == 'emb_cnn' or self.feat_extractor ==
            'vgg_cnn'):
            padded_input = self.conv(padded_input)
        sizes = padded_input.size()
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2],
            sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        hyp, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
            input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(hyp, 1, dim=2)
        strs_gold = [''.join([self.id2label[int(x)] for x in gold_seq]) for
            gold_seq in gold]
        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(
                encoder_padded_outputs, beam_width=beam_width, nbest=1, lm=
                lm, lm_rescoring=lm_rescoring, lm_weight=lm_weight,
                c_weight=c_weight)
            if len(strs_hyps) != sizes[0]:
                None
                strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)
        if verbose:
            None
            None
        return _, strs_hyps, strs_gold


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])
        for i in range(N):
            non_pad_mask[(i), input_lengths[i]:] = 0
    if pad_idx is not None:
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value,
        dim_input, dim_inner, dropout=0.1, src_max_length=2500):
        super(Encoder, self).__init__()
        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner
        self.src_max_length = src_max_length
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.input_linear = nn.Linear(dim_input, dim_model)
        self.layer_norm_input = nn.LayerNorm(dim_model)
        self.positional_encoding = PositionalEncoding(dim_model, src_max_length
            )
        self.layers = nn.ModuleList([EncoderLayer(num_heads, dim_model,
            dim_inner, dim_key, dim_value, dropout=dropout) for _ in range(
            num_layers)])

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        return:
            output: B x T x H
        """
        encoder_self_attn_list = []
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=
            input_lengths)
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len
            )
        encoder_output = self.layer_norm_input(self.input_linear(padded_input)
            ) + self.positional_encoding(padded_input)
        for layer in self.layers:
            encoder_output, self_attn = layer(encoder_output, non_pad_mask=
                non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]
        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value,
        dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, dim_model, dim_key,
            dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(dim_model, dim_inner,
            dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(enc_input, enc_input,
            enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        return enc_output, self_attn


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = constant.args.tgt_max_len
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[(i), :xs[i].size(0)] = xs[i]
    return pad


def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'


def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False


def get_word_segments_per_language(seq):
    """
    Get word segments 
    args:
        seq: String
    output:
        word_segments: list of String
    """
    cur_lang = -1
    words = seq.split(' ')
    temp_words = ''
    word_segments = []
    for i in range(len(words)):
        word = words[i]
        if is_contain_chinese_word(word):
            if cur_lang == -1:
                cur_lang = 1
                temp_words = word
            elif cur_lang == 0:
                cur_lang = 1
                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != '':
                    temp_words += ' '
                temp_words += word
        elif cur_lang == -1:
            cur_lang = 0
            temp_words = word
        elif cur_lang == 1:
            cur_lang = 0
            word_segments.append(temp_words)
            temp_words = word
        else:
            if temp_words != '':
                temp_words += ' '
            temp_words += word
    word_segments.append(temp_words)
    return word_segments


def calculate_lm_score(seq, lm, id2label):
    """
    seq: (1, seq_len)
    id2label: map
    """
    seq_str = ''.join(id2label[char.item()] for char in seq[0]).replace(
        constant.PAD_CHAR, '').replace(constant.SOS_CHAR, '').replace(constant
        .EOS_CHAR, '')
    seq_str = seq_str.replace('  ', ' ')
    seq_arr = get_word_segments_per_language(seq_str)
    seq_str = ''
    for i in range(len(seq_arr)):
        if is_contain_chinese_word(seq_arr[i]):
            for char in seq_arr[i]:
                if seq_str != '':
                    seq_str += ' '
                seq_str += char
        else:
            if seq_str != '':
                seq_str += ' '
            seq_str += seq_arr[i]
    seq_str = seq_str.replace('  ', ' ').replace('  ', ' ')
    if seq_str == '':
        return -999, 0, 0
    score, oov_token = lm.evaluate(seq_str)
    return -1 * score / len(seq_str.split()) + 1, len(seq_str.split()
        ) + 1, oov_token


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.
        device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """
    For masking out the padding part of key sequence.
    """
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask


class Decoder(nn.Module):
    """
    Decoder Layer Transformer class
    """

    def __init__(self, id2label, num_src_vocab, num_trg_vocab, num_layers,
        num_heads, dim_emb, dim_model, dim_inner, dim_key, dim_value,
        dropout=0.1, trg_max_length=1000, emb_trg_sharing=False):
        super(Decoder, self).__init__()
        self.sos_id = constant.SOS_TOKEN
        self.eos_id = constant.EOS_TOKEN
        self.id2label = id2label
        self.num_src_vocab = num_src_vocab
        self.num_trg_vocab = num_trg_vocab
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.dim_model = dim_model
        self.dim_inner = dim_inner
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout_rate = dropout
        self.emb_trg_sharing = emb_trg_sharing
        self.trg_max_length = trg_max_length
        self.trg_embedding = nn.Embedding(num_trg_vocab, dim_emb,
            padding_idx=constant.PAD_TOKEN)
        self.positional_encoding = PositionalEncoding(dim_model, max_length
            =trg_max_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(dim_model, dim_inner,
            num_heads, dim_key, dim_value, dropout=dropout) for _ in range(
            num_layers)])
        self.output_linear = nn.Linear(dim_model, num_trg_vocab, bias=False)
        nn.init.xavier_normal_(self.output_linear.weight)
        if emb_trg_sharing:
            self.output_linear.weight = self.trg_embedding.weight
            self.x_logit_scale = dim_model ** -0.5
        else:
            self.x_logit_scale = 1.0

    def preprocess(self, padded_input):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != constant.PAD_TOKEN] for y in padded_input]
        eos = seq[0].new([self.eos_id])
        sos = seq[0].new([self.sos_id])
        seq_in = [torch.cat([sos, y], dim=0) for y in seq]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad = pad_list(seq_in, self.eos_id)
        seq_out_pad = pad_list(seq_out, constant.PAD_TOKEN)
        assert seq_in_pad.size() == seq_out_pad.size()
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, encoder_padded_outputs,
        encoder_input_lengths):
        """
        args:
            padded_input: B x T
            encoder_padded_outputs: B x T x H
            encoder_input_lengths: B
        returns:
            pred: B x T x vocab
            gold: B x T
        """
        decoder_self_attn_list, decoder_encoder_attn_list = [], []
        seq_in_pad, seq_out_pad = self.preprocess(padded_input)
        non_pad_mask = get_non_pad_mask(seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask_subseq = get_subsequent_mask(seq_in_pad)
        self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=seq_in_pad,
            seq_q=seq_in_pad, pad_idx=constant.EOS_TOKEN)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)
        output_length = seq_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
            encoder_input_lengths, output_length)
        decoder_output = self.dropout(self.trg_embedding(seq_in_pad) * self
            .x_logit_scale + self.positional_encoding(seq_in_pad))
        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_enc_attn = layer(
                decoder_output, encoder_padded_outputs, non_pad_mask=
                non_pad_mask, self_attn_mask=self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            decoder_self_attn_list += [decoder_self_attn]
            decoder_encoder_attn_list += [decoder_enc_attn]
        seq_logit = self.output_linear(decoder_output)
        pred, gold = seq_logit, seq_out_pad
        return pred, gold, decoder_self_attn_list, decoder_encoder_attn_list

    def post_process_hyp(self, hyp):
        """
        args: 
            hyp: list of hypothesis
        output:
            list of hypothesis (string)>
        """
        return ''.join([self.id2label[int(x)] for x in hyp['yseq'][1:]])

    def greedy_search(self, encoder_padded_outputs, beam_width=2,
        lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1):
        """
        Greedy search, decode 1-best utterance
        args:
            encoder_padded_outputs: B x T x H
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        max_seq_len = self.trg_max_length
        ys = torch.ones(encoder_padded_outputs.size(0), 1).fill_(constant.
            SOS_TOKEN).long()
        if constant.args.cuda:
            ys = ys
        decoded_words = []
        for t in range(300):
            non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)
            self_attn_mask = get_subsequent_mask(ys)
            decoder_output = self.dropout(self.trg_embedding(ys) * self.
                x_logit_scale + self.positional_encoding(ys))
            for layer in self.layers:
                decoder_output, _, _ = layer(decoder_output,
                    encoder_padded_outputs, non_pad_mask=non_pad_mask,
                    self_attn_mask=self_attn_mask, dec_enc_attn_mask=None)
            prob = self.output_linear(decoder_output)
            if lm_rescoring:
                local_scores = F.log_softmax(prob, dim=1)
                local_best_scores, local_best_ids = torch.topk(local_scores,
                    beam_width, dim=1)
                best_score = -1
                best_word = None
                for j in range(beam_width):
                    cur_seq = ' '.join(word for word in decoded_words)
                    lm_score, num_words, oov_token = calculate_lm_score(cur_seq
                        , lm, self.id2label)
                    score = local_best_scores[0, j] + lm_score
                    if best_score < score:
                        best_score = score
                        best_word = local_best_ids[0, j]
                        next_word = best_word.unsqueeze(-1)
                decoded_words.append(self.id2label[int(best_word)])
            else:
                _, next_word = torch.max(prob[:, (-1)], dim=1)
                decoded_words.append([(constant.EOS_CHAR if ni.item() ==
                    constant.EOS_TOKEN else self.id2label[ni.item()]) for
                    ni in next_word.view(-1)])
                next_word = next_word.unsqueeze(-1)
            if constant.args.cuda:
                ys = torch.cat([ys, next_word], dim=1)
                ys = ys
            else:
                ys = torch.cat([ys, next_word], dim=1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == constant.EOS_CHAR:
                    break
                else:
                    st += e
            sent.append(st)
        return sent

    def beam_search(self, encoder_padded_outputs, beam_width=2, nbest=5,
        lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1, prob_weight=1.0
        ):
        """
        Beam search, decode nbest utterances
        args:
            encoder_padded_outputs: B x T x H
            beam_size: int
            nbest: int
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        batch_size = encoder_padded_outputs.size(0)
        max_len = encoder_padded_outputs.size(1)
        batch_ids_nbest_hyps = []
        batch_strs_nbest_hyps = []
        for x in range(batch_size):
            encoder_output = encoder_padded_outputs[x].unsqueeze(0)
            ys = torch.ones(1, 1).fill_(constant.SOS_TOKEN).type_as(
                encoder_output).long()
            hyp = {'score': 0.0, 'yseq': ys}
            hyps = [hyp]
            ended_hyps = []
            for i in range(300):
                hyps_best_kept = []
                for hyp in hyps:
                    ys = hyp['yseq']
                    non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)
                    self_attn_mask = get_subsequent_mask(ys)
                    decoder_output = self.dropout(self.trg_embedding(ys) *
                        self.x_logit_scale + self.positional_encoding(ys))
                    for layer in self.layers:
                        decoder_output, _, _ = layer(decoder_output,
                            encoder_output, non_pad_mask=non_pad_mask,
                            self_attn_mask=self_attn_mask,
                            dec_enc_attn_mask=None)
                    seq_logit = self.output_linear(decoder_output[:, (-1)])
                    local_scores = F.log_softmax(seq_logit, dim=1)
                    local_best_scores, local_best_ids = torch.topk(local_scores
                        , beam_width, dim=1)
                    for j in range(beam_width):
                        new_hyp = {}
                        new_hyp['score'] = hyp['score'] + local_best_scores[
                            0, j]
                        new_hyp['yseq'] = torch.ones(1, 1 + ys.size(1)
                            ).type_as(encoder_output).long()
                        new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq'].cpu()
                        new_hyp['yseq'][:, (ys.size(1))] = int(local_best_ids
                            [0, j])
                        hyps_best_kept.append(new_hyp)
                    hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x
                        ['score'], reverse=True)[:beam_width]
                hyps = hyps_best_kept
                if i == max_len - 1:
                    for hyp in hyps:
                        hyp['yseq'] = torch.cat([hyp['yseq'], torch.ones(1,
                            1).fill_(constant.EOS_TOKEN).type_as(
                            encoder_output).long()], dim=1)
                unended_hyps = []
                for hyp in hyps:
                    if hyp['yseq'][0, -1] == constant.EOS_TOKEN:
                        if lm_rescoring:
                            hyp['lm_score'], hyp['num_words'
                                ], oov_token = calculate_lm_score(hyp[
                                'yseq'], lm, self.id2label)
                            num_words = hyp['num_words']
                            hyp['lm_score'] -= oov_token * 2
                            hyp['final_score'] = hyp['score'
                                ] + lm_weight * hyp['lm_score'] + math.sqrt(
                                num_words) * c_weight
                        else:
                            seq_str = ''.join(self.id2label[char.item()] for
                                char in hyp['yseq'][0]).replace(constant.
                                PAD_CHAR, '').replace(constant.SOS_CHAR, ''
                                ).replace(constant.EOS_CHAR, '')
                            seq_str = seq_str.replace('  ', ' ')
                            num_words = len(seq_str.split())
                            hyp['final_score'] = hyp['score'] + math.sqrt(
                                num_words) * c_weight
                        ended_hyps.append(hyp)
                    else:
                        unended_hyps.append(hyp)
                hyps = unended_hyps
                if len(hyps) == 0:
                    break
            num_nbest = min(len(ended_hyps), nbest)
            nbest_hyps = sorted(ended_hyps, key=lambda x: x['final_score'],
                reverse=True)[:num_nbest]
            a_nbest_hyps = sorted(ended_hyps, key=lambda x: x['final_score'
                ], reverse=True)[:beam_width]
            if lm_rescoring:
                for hyp in a_nbest_hyps:
                    seq_str = ''.join(self.id2label[char.item()] for char in
                        hyp['yseq'][0]).replace(constant.PAD_CHAR, '').replace(
                        constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    seq_str = seq_str.replace('  ', ' ')
                    num_words = len(seq_str.split())
            for hyp in nbest_hyps:
                hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
                hyp_strs = self.post_process_hyp(hyp)
                batch_ids_nbest_hyps.append(hyp['yseq'])
                batch_strs_nbest_hyps.append(hyp_strs)
        return batch_ids_nbest_hyps, batch_strs_nbest_hyps


class DecoderLayer(nn.Module):
    """
    Decoder Transformer class
    """

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value,
        dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, dim_model, dim_key,
            dim_value, dropout=dropout)
        self.encoder_attn = MultiHeadAttention(num_heads, dim_model,
            dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(dim_model, dim_inner,
            dropout=dropout)

    def forward(self, decoder_input, encoder_output, non_pad_mask=None,
        self_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, decoder_self_attn = self.self_attn(decoder_input,
            decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output *= non_pad_mask
        decoder_output, decoder_encoder_attn = self.encoder_attn(decoder_output
            , encoder_output, encoder_output, mask=dec_enc_attn_mask)
        decoder_output *= non_pad_mask
        decoder_output = self.pos_ffn(decoder_output)
        decoder_output *= non_pad_mask
        return decoder_output, decoder_self_attn, decoder_encoder_attn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """

    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math
            .log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Layer class
    FFN(x) = max(0, xW1 + b1) W2+ b2
    """

    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        """
        args:
            x: tensor
        output:
            y: tensor
        """
        residual = x
        output = self.dropout(self.linear_2(F.relu(self.linear_1(x))))
        output = self.layer_norm(output + residual)
        return output


class PositionwiseFeedForwardWithConv(nn.Module):
    """
    Position-wise Feedforward Layer Implementation with Convolution class
    """

    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super(PositionwiseFeedForwardWithConv, self).__init__()
        self.conv_1 = nn.Conv1d(dim_model, dim_hidden, 1)
        self.conv_2 = nn.Conv1d(dim_hidden, dim_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.conv_2(F.relu(self.conv_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_model, dim_key, dim_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.value_linear = nn.Linear(dim_model, num_heads * dim_value)
        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 /
            (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (
            self.dim_model + self.dim_key)))
        nn.init.normal_(self.value_linear.weight, mean=0, std=np.sqrt(2.0 /
            (self.dim_model + self.dim_value)))
        self.attention = ScaledDotProductAttention(temperature=np.power(
            dim_key, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.output_linear = nn.Linear(num_heads * dim_value, dim_model)
        nn.init.xavier_normal_(self.output_linear.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: B x T_Q x H, key: B x T_K x H, value: B x T_V x H
        mask: B x T x T (attention mask)
        """
        batch_size, len_query, _ = query.size()
        batch_size, len_key, _ = key.size()
        batch_size, len_value, _ = value.size()
        residual = query
        query = self.query_linear(query).view(batch_size, len_query, self.
            num_heads, self.dim_key)
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads,
            self.dim_key)
        value = self.value_linear(value).view(batch_size, len_value, self.
            num_heads, self.dim_value)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query,
            self.dim_key)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.
            dim_key)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_value,
            self.dim_value)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        output, attn = self.attention(query, key, value, mask=mask)
        output = output.view(self.num_heads, batch_size, len_query, self.
            dim_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size,
            len_query, -1)
        output = self.dropout(self.output_linear(output))
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """

        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class DotProductAttention(nn.Module):
    """
    Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(attention_scores.view(-1,
            input_lengths), dim=1).view(batch_size, -1, input_lengths)
        attention_output = torch.bmm(attention_distribution, values)
        return attention_output, attention_distribution


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
        tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=
                dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[
                    rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                    )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=
                nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
            output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gentaiscool_end2end_asr_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(DotProductAttention(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Encoder(*[], **{'num_layers': 1, 'num_heads': 4, 'dim_model': 4, 'dim_key': 4, 'dim_value': 4, 'dim_input': 4, 'dim_inner': 4}), [torch.rand([4, 4, 4]), [4, 4, 4, 4]], {})

    @_fails_compile()
    def test_002(self):
        self._check(MultiHeadAttention(*[], **{'num_heads': 4, 'dim_model': 4, 'dim_key': 4, 'dim_value': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(PositionalEncoding(*[], **{'dim_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(PositionwiseFeedForward(*[], **{'dim_model': 4, 'dim_ff': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(PositionwiseFeedForwardWithConv(*[], **{'dim_model': 4, 'dim_hidden': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ScaledDotProductAttention(*[], **{'temperature': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

