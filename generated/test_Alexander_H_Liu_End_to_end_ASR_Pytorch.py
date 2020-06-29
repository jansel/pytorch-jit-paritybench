import sys
_module = sys.modules[__name__]
del sys
bin = _module
test_asr = _module
train_asr = _module
train_lm = _module
librispeech = _module
eval = _module
main = _module
src = _module
asr = _module
audio = _module
bert_embedding = _module
ctc = _module
data = _module
decode = _module
lm = _module
module = _module
optim = _module
option = _module
plugin = _module
solver = _module
text = _module
util = _module
test_audio = _module
test_text = _module
generate_vocab_file = _module

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


import math


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions.categorical import Categorical


from functools import partial


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pad_sequence


from torch import nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import abc


from torch.utils.tensorboard import SummaryWriter


def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.0)
    return bias


def init_weights(module):
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                data.zero_()
            elif data.dim() == 2:
                n = data.size(1)
                stdv = 1.0 / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1.0 / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError


class ASR(nn.Module):
    """ ASR model, including Encoder/Decoder(s)"""

    def __init__(self, input_size, vocab_size, init_adadelta, ctc_weight,
        encoder, attention, decoder, emb_drop=0.0):
        super(ASR, self).__init__()
        assert 0 <= ctc_weight <= 1
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None
        self.encoder = Encoder(input_size, **encoder)
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(self.encoder.out_dim, vocab_size)
        if self.enable_att:
            self.dec_dim = decoder['dim']
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            self.embed_drop = nn.Dropout(emb_drop)
            self.decoder = Decoder(self.encoder.out_dim + self.dec_dim,
                vocab_size, **decoder)
            query_dim = self.dec_dim * self.decoder.layer
            self.attention = Attention(self.encoder.out_dim, query_dim, **
                attention)
        if init_adadelta:
            self.apply(init_weights)
            for l in range(self.decoder.layer):
                bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
                bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        """ Setting up all memory states for beam decoding"""
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def create_msg(self):
        msg = []
        msg.append(
            "Model spec.| Encoder's downsampling rate of time axis is {}.".
            format(self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append(
                '           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.'
                )
        if self.encoder.cnn:
            msg.append(
                '           | CNN Extractor w/ time downsampling rate = 4 in encoder enabled.'
                )
        if self.enable_ctc:
            msg.append(
                '           | CTC training on encoder enabled ( lambda = {}).'
                .format(self.ctc_weight))
        if self.enable_att:
            msg.append(
                '           | {} attention decoder enabled ( lambda = {}).'
                .format(self.attention.mode, 1 - self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len, decode_step, tf_rate=0.0,
        teacher=None, emb_decoder=None, get_dec_state=False):
        """
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape 
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxL] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        """
        bs = audio_feature.shape[0]
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None
        encode_feature, encode_len = self.encoder(audio_feature, feature_len)
        if self.enable_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature), dim=-1)
        if self.enable_att:
            self.decoder.init_state(bs)
            self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros(bs, dtype=torch.long,
                device=encode_feature.device))
            att_seq, output_seq = [], []
            if teacher is not None:
                teacher = self.embed_drop(self.pre_embed(teacher))
            for t in range(decode_step):
                attn, context = self.attention(self.decoder.get_query(),
                    encode_feature, encode_len)
                decoder_input = torch.cat([last_char, context], dim=-1)
                cur_char, d_state = self.decoder(decoder_input)
                if teacher is not None:
                    if tf_rate == 1 or torch.rand(1).item() <= tf_rate:
                        last_char = teacher[:, (t), :]
                    else:
                        with torch.no_grad():
                            if (emb_decoder is not None and emb_decoder.
                                apply_fuse):
                                _, cur_prob = emb_decoder(d_state, cur_char,
                                    return_loss=False)
                            else:
                                cur_prob = cur_char.softmax(dim=-1)
                            sampled_char = Categorical(cur_prob).sample()
                        last_char = self.embed_drop(self.pre_embed(
                            sampled_char))
                else:
                    if emb_decoder is not None and emb_decoder.apply_fuse:
                        _, cur_char = emb_decoder(d_state, cur_char,
                            return_loss=False)
                    last_char = self.pre_embed(torch.argmax(cur_char, dim=-1))
                output_seq.append(cur_char)
                att_seq.append(attn)
                if get_dec_state:
                    dec_state.append(d_state)
            att_output = torch.stack(output_seq, dim=1)
            att_seq = torch.stack(att_seq, dim=2)
            if get_dec_state:
                dec_state = torch.stack(dec_state, dim=1)
        return ctc_output, encode_len, att_output, att_seq, dec_state


class Decoder(nn.Module):
    """ Decoder (a.k.a. Speller in LAS) """

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'
        self.layers = getattr(nn, module)(input_dim, dim, num_layers=layer,
            dropout=dropout, batch_first=True)
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        """ Set all hidden states to zeros """
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = torch.zeros((self.layer, bs, self.dim),
                device=device), torch.zeros((self.layer, bs, self.dim),
                device=device)
        else:
            self.hidden_state = torch.zeros((self.layer, bs, self.dim),
                device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        """ Set all hidden states/cells, for decoding purpose"""
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = hidden_state[0], hidden_state[1]
        else:
            self.hidden_state = hidden_state

    def get_state(self):
        """ Return all hidden states/cells, for decoding purpose"""
        if self.enable_cell:
            return self.hidden_state[0].cpu(), self.hidden_state[1].cpu()
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        """ Return state of all layers as query for attention """
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.
                dim * self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim *
                self.layer)

    def forward(self, x):
        """ Decode and transform into vocab """
        if not self.training:
            self.layers.flatten_parameters()
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)
        char = self.char_trans(self.final_dropout(x))
        return char, x


class Encoder(nn.Module):
    """ Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details."""

    def __init__(self, input_size, prenet, module, bidirection, dim,
        dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()
        self.vgg = prenet == 'vgg'
        self.cnn = prenet == 'cnn'
        self.sample_rate = 1
        assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
        assert len(dropout) == len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'
        module_list = []
        input_dim = input_size
        if self.vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate * 4
        if self.cnn:
            cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
            module_list.append(cnn_extractor)
            input_dim = cnn_extractor.out_dim
            self.sample_rate = self.sample_rate * 4
        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l],
                    bidirection, dropout[l], layer_norm[l], sample_rate[l],
                    sample_style, proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate * sample_rate[l]
        else:
            raise NotImplementedError
        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        return input_x, enc_len


class CMVN(torch.jit.ScriptModule):
    __constants__ = ['mode', 'dim', 'eps']

    def __init__(self, mode='global', dim=2, eps=1e-10):
        super(CMVN, self).__init__()
        if mode != 'global':
            raise NotImplementedError(
                'Only support global mean variance normalization.')
        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == 'global':
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std
                (self.dim, keepdim=True))

    def extra_repr(self):
        return 'mode={}, dim={}, eps={}'.format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):
    __constants__ = ['order', 'window_size', 'padding']

    def __init__(self, order=1, window_size=2):
        super(Delta, self).__init__()
        self.order = order
        self.window_size = window_size
        filters = self._create_filters(order, window_size)
        self.register_buffer('filters', filters)
        self.padding = 0, (filters.shape[-1] - 1) // 2

    @torch.jit.script_method
    def forward(self, x):
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i - 1]) - 1) // 2
            curr_offset = prev_offset + window_size
            curr = [0] * (len(scales[i - 1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j + k + curr_offset] += j * scales[i - 1][k +
                        prev_offset]
            curr = [(x / normalizer) for x in curr]
            scales.append(curr)
        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding
        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return 'order={}, window_size={}'.format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, x):
        x = x.permute(2, 0, 1)
        return x.reshape(x.size(0), -1).detach()


class ExtractAudioFeature(nn.Module):

    def __init__(self, mode='fbank', num_mel_bins=40, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        self.extract_fn = (torchaudio.compliance.kaldi.fbank if mode ==
            'fbank' else torchaudio.compliance.kaldi.mfcc)
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)
        y = self.extract_fn(waveform, num_mel_bins=self.num_mel_bins,
            channel=-1, sample_frequency=sample_rate, **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return 'mode={}, num_mel_bins={}'.format(self.mode, self.num_mel_bins)


def generate_embedding(bert_model, labels):
    """Generate bert's embedding from fine-tuned model."""
    batch_size, time = labels.shape
    cls_ids = torch.full((batch_size, 1), bert_model.bert_text_encoder.
        cls_idx, dtype=labels.dtype, device=labels.device)
    bert_labels = torch.cat([cls_ids, labels], 1)
    eos_idx = bert_model.bert_text_encoder.eos_idx
    sep_idx = bert_model.bert_text_encoder.sep_idx
    bert_labels[bert_labels == eos_idx] = sep_idx
    embedding, _ = bert_model.bert(bert_labels, output_all_encoded_layers=True)
    embedding = torch.stack(embedding).sum(0)
    embedding = embedding[:, 1:]
    assert labels.shape == embedding.shape[:-1]
    return embedding


class BertLikeSentencePieceTextEncoder(object):

    def __init__(self, text_encoder):
        if not isinstance(text_encoder, text.SubwordTextEncoder):
            raise TypeError(
                '`text_encoder` must be an instance of `src.text.SubwordTextEncoder`.'
                )
        self.text_encoder = text_encoder

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size + 3

    @property
    def cls_idx(self):
        return self.vocab_size - 3

    @property
    def sep_idx(self):
        return self.vocab_size - 2

    @property
    def mask_idx(self):
        return self.vocab_size - 1

    @property
    def eos_idx(self):
        return self.text_encoder.eos_idx


def load_fine_tuned_model(bert_model, text_encoder, path):
    """Load fine-tuned bert model given text encoder and checkpoint path."""
    bert_text_encoder = BertLikeSentencePieceTextEncoder(text_encoder)
    model = BertForMaskedLM.from_pretrained(bert_model)
    model.bert_text_encoder = bert_text_encoder
    model.bert.embeddings.word_embeddings = nn.Embedding(bert_text_encoder.
        vocab_size, model.bert.embeddings.word_embeddings.weight.shape[1])
    model.config.vocab_size = bert_text_encoder.vocab_size
    model.cls = BertOnlyMLMHead(model.config, model.bert.embeddings.
        word_embeddings.weight)
    model.load_state_dict(torch.load(path))
    return model


class BertEmbeddingPredictor(nn.Module):

    def __init__(self, bert_model, text_encoder, path):
        super(BertEmbeddingPredictor, self).__init__()
        self.model = load_fine_tuned_model(bert_model, text_encoder, path)

    def forward(self, labels):
        self.eval()
        return generate_embedding(self.model, labels)


class CTCPrefixScore:
    """ 
    CTC Prefix score calculator
    An implementation of Algo. 2 in https://www.merl.com/publications/docs/TR2017-190.pdf (Watanabe et. al.)
    Reference (official implementation): https://github.com/espnet/espnet/tree/master/espnet/nets
    """

    def __init__(self, x):
        self.logzero = -100000000.0
        self.blank = 0
        self.eos = 1
        self.x = x.cpu().numpy()[0]
        self.odim = x.shape[-1]
        self.input_length = len(self.x)

    def init_state(self):
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def full_compute(self, g, r_prev):
        """Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function computes all possible tokens for c (memory inefficient)"""
        prefix_length = len(g)
        last_char = g[-1] if prefix_length > 0 else 0
        r = np.full((self.input_length, 2, self.odim), self.logzero, dtype=
            np.float32)
        start = max(1, prefix_length)
        if prefix_length == 0:
            r[(0), (0), :] = self.x[(0), :]
        psi = r[(start - 1), (0), :]
        phi = np.logaddexp(r_prev[:, (0)], r_prev[:, (1)])
        for t in range(start, self.input_length):
            prev_blank = np.full(self.odim, r_prev[t - 1, 1], dtype=np.float32)
            prev_nonblank = np.full(self.odim, r_prev[t - 1, 0], dtype=np.
                float32)
            prev_nonblank[last_char] = self.logzero
            phi = np.logaddexp(prev_nonblank, prev_blank)
            r[(t), (0), :] = np.logaddexp(r[(t - 1), (0), :], phi) + self.x[(
                t), :]
            r[(t), (1), :] = np.logaddexp(r[(t - 1), (1), :], r[(t - 1), (0
                ), :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi + self.x[(t), :])
        return psi, np.rollaxis(r, 2)

    def cheap_compute(self, g, r_prev, candidates):
        """Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function considers only those tokens in candidates for c (memory efficient)"""
        prefix_length = len(g)
        odim = len(candidates)
        last_char = g[-1] if prefix_length > 0 else 0
        r = np.full((self.input_length, 2, len(candidates)), self.logzero,
            dtype=np.float32)
        start = max(1, prefix_length)
        if prefix_length == 0:
            r[(0), (0), :] = self.x[0, candidates]
        psi = r[(start - 1), (0), :]
        sum_prev = np.logaddexp(r_prev[:, (0)], r_prev[:, (1)])
        phi = np.repeat(sum_prev[..., None], odim, axis=-1)
        if prefix_length > 0 and last_char in candidates:
            phi[:, (candidates.index(last_char))] = r_prev[:, (1)]
        for t in range(start, self.input_length):
            r[(t), (0), :] = np.logaddexp(r[(t - 1), (0), :], phi[t - 1]
                ) + self.x[t, candidates]
            r[(t), (1), :] = np.logaddexp(r[(t - 1), (1), :], r[(t - 1), (0
                ), :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi[t - 1,] + self.x[t, candidates])
        if self.eos in candidates:
            psi[candidates.index(self.eos)] = sum_prev[-1]
        return psi, np.rollaxis(r, 2)


CTC_BEAM_RATIO = 1.5


class Hypothesis:
    """Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)"""

    def __init__(self, decoder_state, output_seq, output_scores, lm_state,
        ctc_state, ctc_prob, att_map):
        assert len(output_seq) == len(output_scores)
        self.decoder_state = decoder_state
        self.att_map = att_map
        if type(lm_state) is tuple:
            self.lm_state = lm_state[0].cpu(), lm_state[1].cpu()
        elif lm_state is None:
            self.lm_state = None
        else:
            self.lm_state = lm_state.cpu()
        self.output_seq = output_seq
        self.output_scores = output_scores
        self.ctc_state = ctc_state
        self.ctc_prob = ctc_prob

    def avgScore(self):
        """Return the averaged log probability of hypothesis"""
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None, lm_state=
        None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        """Expand current hypothesis with a given beam size"""
        new_hypothesis = []
        term_score = None
        ctc_s, ctc_p = None, None
        beam_size = topi.shape[-1]
        for i in range(beam_size):
            if topi[i].item() == 1:
                term_score = topv[i].cpu()
                continue
            idxes = self.output_seq[:]
            scores = self.output_scores[:]
            idxes.append(topi[i].cpu())
            scores.append(topv[i].cpu())
            if ctc_state is not None:
                idx = ctc_candidates.index(topi[i].item())
                ctc_s = ctc_state[(idx), :, :]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state, output_seq=
                idxes, output_scores=scores, lm_state=lm_state, ctc_state=
                ctc_s, ctc_prob=ctc_p, att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    def get_state(self, device):
        prev_token = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        prev_token = torch.LongTensor([prev_token]).to(device)
        att_map = self.att_map.to(device) if self.att_map is not None else None
        if type(self.lm_state) is tuple:
            lm_state = self.lm_state[0].to(device), self.lm_state[1].to(device)
        elif self.lm_state is None:
            lm_state = None
        else:
            lm_state = self.lm_state.to(device)
        return (prev_token, self.decoder_state, att_map, lm_state, self.
            ctc_state)

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]


LOG_ZERO = -10000000.0


class BeamDecoder(nn.Module):
    """ Beam decoder for ASR """

    def __init__(self, asr, emb_decoder, beam_size, min_len_ratio,
        max_len_ratio, lm_path='', lm_config='', lm_weight=0.0, ctc_weight=0.0
        ):
        super().__init__()
        self.beam_size = beam_size
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.asr = asr
        assert self.asr.enable_att
        self.apply_ctc = ctc_weight > 0
        if self.apply_ctc:
            assert self.asr.ctc_weight > 0, 'ASR was not trained with CTC decoder'
            self.ctc_w = ctc_weight
            self.ctc_beam_size = int(CTC_BEAM_RATIO * self.beam_size)
        self.apply_lm = lm_weight > 0
        if self.apply_lm:
            self.lm_w = lm_weight
            self.lm_path = lm_path
            lm_config = yaml.load(open(lm_config, 'r'), Loader=yaml.FullLoader)
            self.lm = RNNLM(self.asr.vocab_size, **lm_config['model'])
            self.lm.load_state_dict(torch.load(self.lm_path, map_location=
                'cpu')['model'])
            self.lm.eval()
        self.apply_emb = emb_decoder is not None
        if self.apply_emb:
            self.emb_decoder = emb_decoder

    def create_msg(self):
        msg = ['Decode spec| Beam size = {}\t| Min/Max len ratio = {}/{}'.
            format(self.beam_size, self.min_len_ratio, self.max_len_ratio)]
        if self.apply_ctc:
            msg.append(
                '           |Joint CTC decoding enabled \t| weight = {:.2f}\t'
                .format(self.ctc_w))
        if self.apply_lm:
            msg.append(
                '           |Joint LM decoding enabled \t| weight = {:.2f}\t| src = {}'
                .format(self.lm_w, self.lm_path))
        if self.apply_emb:
            msg.append(
                '           |Joint Emb. decoding enabled \t| weight = {:.2f}'
                .format(self.lm_w, self.emb_decoder.fuse_lambda.mean().cpu(
                ).item()))
        return msg

    def forward(self, audio_feature, feature_len):
        assert audio_feature.shape[0
            ] == 1, 'Batchsize == 1 is required for beam search'
        batch_size = audio_feature.shape[0]
        device = audio_feature.device
        dec_state = self.asr.decoder.init_state(batch_size)
        self.asr.attention.reset_mem()
        max_output_len = int(np.ceil(feature_len.cpu().item() * self.
            max_len_ratio))
        min_output_len = int(np.ceil(feature_len.cpu().item() * self.
            min_len_ratio))
        store_att = self.asr.attention.mode == 'loc'
        prev_token = torch.zeros((batch_size, 1), dtype=torch.long, device=
            device)
        final_hypothesis, next_top_hypothesis = [], []
        ctc_state, ctc_prob, candidates, lm_state = None, None, None, None
        encode_feature, encode_len = self.asr.encoder(audio_feature,
            feature_len)
        if self.apply_ctc:
            ctc_output = F.log_softmax(self.asr.ctc_layer(encode_feature),
                dim=-1)
            ctc_prefix = CTCPrefixScore(ctc_output)
            ctc_state = ctc_prefix.init_state()
        prev_top_hypothesis = [Hypothesis(decoder_state=dec_state,
            output_seq=[], output_scores=[], lm_state=None, ctc_prob=0,
            ctc_state=ctc_state, att_map=None)]
        for t in range(max_output_len):
            for hypothesis in prev_top_hypothesis:
                (prev_token, prev_dec_state, prev_attn, prev_lm_state,
                    prev_ctc_state) = hypothesis.get_state(device)
                self.asr.set_state(prev_dec_state, prev_attn)
                attn, context = self.asr.attention(self.asr.decoder.
                    get_query(), encode_feature, encode_len)
                asr_prev_token = self.asr.pre_embed(prev_token)
                decoder_input = torch.cat([asr_prev_token, context], dim=-1)
                cur_prob, d_state = self.asr.decoder(decoder_input)
                if self.apply_emb:
                    _, cur_prob = self.emb_decoder(d_state, cur_prob,
                        return_loss=False)
                else:
                    cur_prob = F.log_softmax(cur_prob, dim=-1)
                if self.apply_ctc:
                    _, ctc_candidates = cur_prob.squeeze(0).topk(self.
                        ctc_beam_size, dim=-1)
                    candidates = ctc_candidates.cpu().tolist()
                    ctc_prob, ctc_state = ctc_prefix.cheap_compute(hypothesis
                        .outIndex, prev_ctc_state, candidates)
                    ctc_char = torch.FloatTensor(ctc_prob - hypothesis.ctc_prob
                        )
                    hack_ctc_char = torch.zeros_like(cur_prob).data.fill_(
                        LOG_ZERO)
                    for idx, char in enumerate(candidates):
                        hack_ctc_char[0, char] = ctc_char[idx]
                    cur_prob = (1 - self.ctc_w
                        ) * cur_prob + self.ctc_w * hack_ctc_char
                    cur_prob[0, 0] = LOG_ZERO
                if self.apply_lm:
                    lm_input = prev_token.unsqueeze(1)
                    lm_output, lm_state = self.lm(lm_input, torch.ones([
                        batch_size]), hidden=prev_lm_state)
                    lm_output = lm_output.squeeze(0)
                    cur_prob += self.lm_w * lm_output.log_softmax(dim=-1)
                topv, topi = cur_prob.squeeze(0).topk(self.beam_size)
                prev_attn = self.asr.attention.att_layer.prev_att.cpu(
                    ) if store_att else None
                final, top = hypothesis.addTopk(topi, topv, self.asr.
                    decoder.get_state(), att_map=prev_attn, lm_state=
                    lm_state, ctc_state=ctc_state, ctc_prob=ctc_prob,
                    ctc_candidates=candidates)
                if final is not None and t >= min_output_len:
                    final_hypothesis.append(final)
                    if self.beam_size == 1:
                        return final_hypothesis
                next_top_hypothesis.extend(top)
            next_top_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)
            prev_top_hypothesis = next_top_hypothesis[:self.beam_size]
            next_top_hypothesis = []
        final_hypothesis += prev_top_hypothesis
        final_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)
        return final_hypothesis[:self.beam_size]


class RNNLM(nn.Module):
    """ RNN Language Model """

    def __init__(self, vocab_size, emb_tying, emb_dim, module, dim,
        n_layers, dropout):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.emb_tying = emb_tying
        if emb_tying:
            assert emb_dim == dim, 'Output dim of RNN should be identical to embedding if using weight tying.'
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.rnn = getattr(nn, module.upper())(emb_dim, dim, num_layers=
            n_layers, dropout=dropout, batch_first=True)
        if not self.emb_tying:
            self.trans = nn.Linear(emb_dim, vocab_size)

    def create_msg(self):
        msg = [
            'Model spec.| RNNLM weight tying = {}, # of layers = {}, dim = {}'
            .format(self.emb_tying, self.n_layers, self.dim)]
        return msg

    def forward(self, x, lens, hidden=None):
        emb_x = self.dp1(self.emb(x))
        if not self.training:
            self.rnn.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(emb_x, lens, batch_first
            =True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True
            )
        if self.emb_tying:
            outputs = F.linear(self.dp2(outputs), self.emb.weight)
        else:
            outputs = self.trans(self.dp2(outputs))
        return outputs, hidden


class VGGExtractor(nn.Module):
    """ VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf"""

    def __init__(self, input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.extractor = nn.Sequential(nn.Conv2d(in_channel, self.init_dim,
            3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(self.init_dim,
            self.init_dim, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d
            (2, stride=2), nn.Conv2d(self.init_dim, self.hide_dim, 3,
            stride=1, padding=1), nn.ReLU(), nn.Conv2d(self.hide_dim, self.
            hide_dim, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2,
            stride=2))

    def check_dim(self, input_dim):
        if input_dim % 13 == 0:
            return int(input_dim / 13), 13, 13 // 4 * self.hide_dim
        elif input_dim % 40 == 0:
            return int(input_dim / 40), 40, 40 // 4 * self.hide_dim
        else:
            raise ValueError(
                'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '
                 + input_dim)

    def view_input(self, feature, feat_len):
        feat_len = feat_len // 4
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)
        return feature, feat_len

    def forward(self, feature, feat_len):
        feature, feat_len = self.view_input(feature, feat_len)
        feature = self.extractor(feature)
        feature = feature.transpose(1, 2)
        feature = feature.contiguous().view(feature.shape[0], feature.shape
            [1], self.out_dim)
        return feature, feat_len


class CNNExtractor(nn.Module):
    """ A simple 2-layer CNN extractor for acoustic feature down-sampling"""

    def __init__(self, input_dim, out_dim):
        super(CNNExtractor, self).__init__()
        self.out_dim = out_dim
        self.extractor = nn.Sequential(nn.Conv1d(input_dim, out_dim, 4,
            stride=2, padding=1), nn.Conv1d(out_dim, out_dim, 4, stride=2,
            padding=1))

    def forward(self, feature, feat_len):
        feat_len = feat_len // 4
        feature = feature.transpose(1, 2)
        feature = self.extractor(feature)
        feature = feature.transpose(1, 2)
        return feature, feat_len


class RNNLayer(nn.Module):
    """ RNN wrapper, includes time-downsampling"""

    def __init__(self, input_dim, module, dim, bidirection, dropout,
        layer_norm, sample_rate, sample_style, proj):
        super(RNNLayer, self).__init__()
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = (sample_rate * rnn_out_dim if sample_rate > 1 and 
            sample_style == 'concat' else rnn_out_dim)
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj
        if self.sample_style not in ['drop', 'concat']:
            raise ValueError('Unsupported Sample Style: ' + self.sample_style)
        self.layer = getattr(nn, module.upper())(input_dim, dim,
            bidirectional=bidirection, num_layers=1, batch_first=True)
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        if not self.training:
            self.layer.flatten_parameters()
        output, _ = self.layer(input_x)
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)
        if self.sample_rate > 1:
            batch_size, timestep, feature_dim = output.shape
            x_len = x_len // self.sample_rate
            if self.sample_style == 'drop':
                output = output[:, ::self.sample_rate, :].contiguous()
            else:
                if timestep % self.sample_rate != 0:
                    output = output[:, :-(timestep % self.sample_rate), :]
                output = output.contiguous().view(batch_size, int(timestep /
                    self.sample_rate), feature_dim * self.sample_rate)
        if self.proj:
            output = torch.tanh(self.pj(output))
        return output, x_len


class BaseAttention(nn.Module):
    """ Base module for attentions """

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[(idx), :, sl:] = 1
        self.mask = torch.from_numpy(self.mask).view(-1, ts)

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(1)
        return output, attn


def load_embedding(text_encoder, embedding_filepath):
    with open(embedding_filepath, 'r') as f:
        vocab_size, embedding_size = [int(x) for x in f.readline().strip().
            split()]
        embeddings = np.zeros((text_encoder.vocab_size, embedding_size))
        unk_count = 0
        for line in f:
            vocab, emb = line.strip().split(' ', 1)
            if vocab == '</s>':
                vocab = '<eos>'
            if text_encoder.token_type == 'subword':
                idx = text_encoder.spm.piece_to_id(vocab)
            else:
                idx = text_encoder.encode(vocab)[0]
            if idx == text_encoder.unk_idx:
                unk_count += 1
                embeddings[idx] += np.asarray([float(x) for x in emb.split(
                    ' ')])
            else:
                embeddings[idx] = np.asarray([float(x) for x in emb.split(' ')]
                    )
        if unk_count != 0:
            embeddings[text_encoder.unk_idx] /= unk_count
        return embeddings


class EmbeddingRegularizer(nn.Module):
    """ Perform word embedding regularization training for ASR"""

    def __init__(self, tokenizer, dec_dim, enable, src, distance, weight,
        fuse, temperature, freeze=True, fuse_normalize=False, dropout=0.0,
        bert=None):
        super(EmbeddingRegularizer, self).__init__()
        self.enable = enable
        if enable:
            if bert is not None:
                self.use_bert = True
                if not isinstance(bert, str):
                    raise ValueError(
                        '`bert` should be a str specifying bert config such as "bert-base-uncased".'
                        )
                self.emb_table = BertEmbeddingPredictor(bert, tokenizer, src)
                vocab_size, emb_dim = (self.emb_table.model.bert.embeddings
                    .word_embeddings.weight.shape)
                vocab_size = vocab_size - 3
                self.dim = emb_dim
            else:
                self.use_bert = False
                pretrained_emb = torch.FloatTensor(load_embedding(tokenizer,
                    src))
                vocab_size, emb_dim = pretrained_emb.shape
                self.dim = emb_dim
                self.emb_table = nn.Embedding.from_pretrained(pretrained_emb,
                    freeze=freeze, padding_idx=0)
            self.emb_net = nn.Sequential(nn.Linear(dec_dim, (emb_dim +
                dec_dim) // 2), nn.ReLU(), nn.Linear((emb_dim + dec_dim) //
                2, emb_dim))
            self.weight = weight
            self.distance = distance
            self.fuse_normalize = fuse_normalize
            if distance == 'CosEmb':
                self.measurement = nn.CosineEmbeddingLoss(reduction='none')
            elif distance == 'MSE':
                self.measurement = nn.MSELoss(reduction='none')
            else:
                raise NotImplementedError
            self.apply_dropout = dropout > 0
            if self.apply_dropout:
                self.dropout = nn.Dropout(dropout)
            self.apply_fuse = fuse != 0
            if self.apply_fuse:
                if fuse == -1:
                    self.fuse_type = 'learnable'
                    self.fuse_learnable = True
                    self.fuse_lambda = nn.Parameter(data=torch.FloatTensor(
                        [0.5]))
                elif fuse == -2:
                    self.fuse_type = 'vocab-wise learnable'
                    self.fuse_learnable = True
                    self.fuse_lambda = nn.Parameter(torch.ones(vocab_size) *
                        0.5)
                else:
                    self.fuse_type = str(fuse)
                    self.fuse_learnable = False
                    self.register_buffer('fuse_lambda', torch.FloatTensor([
                        fuse]))
                if temperature == -1:
                    self.temperature = 'learnable'
                    self.temp = nn.Parameter(data=torch.FloatTensor([1]))
                elif temperature == -2:
                    self.temperature = 'elementwise'
                    self.temp = nn.Parameter(torch.ones(vocab_size))
                else:
                    self.temperature = str(temperature)
                    self.register_buffer('temp', torch.FloatTensor([
                        temperature]))
                self.eps = 1e-08

    def create_msg(self):
        msg = [
            'Plugin.    | Word embedding regularization enabled (type:{}, weight:{})'
            .format(self.distance, self.weight)]
        if self.apply_fuse:
            msg.append(
                '           | Embedding-fusion decoder enabled ( temp. = {}, lambda = {} )'
                .format(self.temperature, self.fuse_type))
        return msg

    def get_weight(self):
        if self.fuse_learnable:
            return torch.sigmoid(self.fuse_lambda).mean().cpu().data
        else:
            return self.fuse_lambda

    def get_temp(self):
        return nn.functional.relu(self.temp).mean()

    def fuse_prob(self, x_emb, dec_logit):
        """ Takes context and decoder logit to perform word embedding fusion """
        if self.fuse_normalize:
            emb_logit = nn.functional.linear(nn.functional.normalize(x_emb,
                dim=-1), nn.functional.normalize(self.emb_table.weight, dim=-1)
                )
        else:
            emb_logit = nn.functional.linear(x_emb, self.emb_table.weight)
        emb_prob = (nn.functional.relu(self.temp) * emb_logit).softmax(dim=-1)
        dec_prob = dec_logit.softmax(dim=-1)
        if self.fuse_learnable:
            fused_prob = (1 - torch.sigmoid(self.fuse_lambda)
                ) * dec_prob + torch.sigmoid(self.fuse_lambda) * emb_prob
        else:
            fused_prob = (1 - self.fuse_lambda
                ) * dec_prob + self.fuse_lambda * emb_prob
        log_fused_prob = (fused_prob + self.eps).log()
        return log_fused_prob

    def forward(self, dec_state, dec_logit, label=None, return_loss=True):
        log_fused_prob = None
        loss = None
        if self.apply_dropout:
            dec_state = self.dropout(dec_state)
        x_emb = self.emb_net(dec_state)
        if return_loss:
            b, t = label.shape
            if self.use_bert:
                with torch.no_grad():
                    y_emb = self.emb_table(label).contiguous()
            else:
                y_emb = self.emb_table(label)
            if self.distance == 'CosEmb':
                loss = self.measurement(x_emb.view(-1, self.dim), y_emb.
                    view(-1, self.dim), torch.ones(1))
            else:
                loss = self.measurement(x_emb.view(-1, self.dim), y_emb.
                    view(-1, self.dim))
            loss = loss.view(b, t)
            loss = torch.where(label != 0, loss, torch.zeros_like(loss))
            loss = torch.mean(loss.sum(dim=-1) / (label != 0).sum(dim=-1).
                float())
        if self.apply_fuse:
            log_fused_prob = self.fuse_prob(x_emb, dec_logit)
        return loss, log_fused_prob


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Alexander_H_Liu_End_to_end_ASR_Pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(CNNExtractor(*[], **{'input_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Decoder(*[], **{'input_dim': 4, 'vocab_size': 4, 'module': LSTM, 'dim': 4, 'layer': 1, 'dropout': 0.5}), [torch.rand([4, 4])], {})

