import sys
_module = sys.modules[__name__]
del sys
beam_search = _module
batcher = _module
config = _module
data = _module
eval = _module
make_data_files = _module
model = _module
train = _module
train_util = _module

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


import torch as T


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.distributions import Categorical


from numpy import random


_global_config['rand_unif_init_mag'] = 4


def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-config.rand_unif_init_mag, config.
                rand_unif_init_mag)
        elif 'bias' in name:
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.0)
            bias.data[start:end].fill_(1.0)


_global_config['trunc_norm_init_std'] = 4


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


_global_config['hidden_dim'] = 4


_global_config['emb_dim'] = 4


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, x, seq_lens):
        packed = pack_padded_sequence(x, seq_lens, batch_first=True)
        enc_out, enc_hid = self.lstm(packed)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = enc_out.contiguous()
        h, c = enc_hid
        h = T.cat(list(h), dim=1)
        c = T.cat(list(c), dim=1)
        h_reduced = F.relu(self.reduce_h(h))
        c_reduced = F.relu(self.reduce_c(c))
        return enc_out, (h_reduced, c_reduced)


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


_global_config['intra_encoder'] = 4


class encoder_attention(nn.Module):

    def __init__(self):
        super(encoder_attention, self).__init__()
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2,
            bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
        """ Perform attention over encoder hidden states
        :param st_hat: decoder hidden state at current time step
        :param h: encoder hidden states
        :param enc_padding_mask:
        :param sum_temporal_srcs: if using intra-temporal attention, contains summation of attention weights from previous decoder time steps
        """
        et = self.W_h(h)
        dec_fea = self.W_s(st_hat).unsqueeze(1)
        et = et + dec_fea
        et = T.tanh(et)
        et = self.v(et).squeeze(2)
        if config.intra_encoder:
            exp_et = T.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et
                sum_temporal_srcs = get_cuda(T.FloatTensor(et.size()).fill_
                    (1e-10)) + exp_et
            else:
                et1 = exp_et / sum_temporal_srcs
                sum_temporal_srcs = sum_temporal_srcs + exp_et
        else:
            et1 = F.softmax(et, dim=1)
        at = et1 * enc_padding_mask
        normalization_factor = at.sum(1, keepdim=True)
        at = at / normalization_factor
        at = at.unsqueeze(1)
        ct_e = T.bmm(at, h)
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)
        return ct_e, at, sum_temporal_srcs


_global_config['intra_decoder'] = 4


class decoder_attention(nn.Module):

    def __init__(self):
        super(decoder_attention, self).__init__()
        if config.intra_decoder:
            self.W_prev = nn.Linear(config.hidden_dim, config.hidden_dim,
                bias=False)
            self.W_s = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.v = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t, prev_s):
        """Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        """
        if config.intra_decoder is False:
            ct_d = get_cuda(T.zeros(s_t.size()))
        elif prev_s is None:
            ct_d = get_cuda(T.zeros(s_t.size()))
            prev_s = s_t.unsqueeze(1)
        else:
            et = self.W_prev(prev_s)
            dec_fea = self.W_s(s_t).unsqueeze(1)
            et = et + dec_fea
            et = T.tanh(et)
            et = self.v(et).squeeze(2)
            at = F.softmax(et, dim=1).unsqueeze(1)
            ct_d = T.bmm(at, prev_s).squeeze(1)
            prev_s = T.cat([prev_s, s_t.unsqueeze(1)], dim=1)
        return ct_d, prev_s


_global_config['vocab_size'] = 4


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.enc_attention = encoder_attention()
        self.dec_attention = decoder_attention()
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim,
            config.emb_dim)
        self.lstm = nn.LSTMCell(config.emb_dim, config.hidden_dim)
        init_lstm_wt(self.lstm)
        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1
            )
        self.V = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.V1 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.V1)

    def forward(self, x_t, s_t, enc_out, enc_padding_mask, ct_e,
        extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s):
        x = self.x_context(T.cat([x_t, ct_e], dim=1))
        s_t = self.lstm(x, s_t)
        dec_h, dec_c = s_t
        st_hat = T.cat([dec_h, dec_c], dim=1)
        ct_e, attn_dist, sum_temporal_srcs = self.enc_attention(st_hat,
            enc_out, enc_padding_mask, sum_temporal_srcs)
        ct_d, prev_s = self.dec_attention(dec_h, prev_s)
        p_gen = T.cat([ct_e, ct_d, st_hat, x], 1)
        p_gen = self.p_gen_linear(p_gen)
        p_gen = T.sigmoid(p_gen)
        out = T.cat([dec_h, ct_e, ct_d], dim=1)
        out = self.V(out)
        out = self.V1(out)
        vocab_dist = F.softmax(out, dim=1)
        vocab_dist = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        if extra_zeros is not None:
            vocab_dist = T.cat([vocab_dist, extra_zeros], dim=1)
        final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab,
            attn_dist_)
        return final_dist, s_t, ct_e, sum_temporal_srcs, prev_s


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embeds.weight)
        self.encoder = get_cuda(self.encoder)
        self.decoder = get_cuda(self.decoder)
        self.embeds = get_cuda(self.embeds)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rohithreddy024_Text_Summarizer_Pytorch(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(decoder_attention(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4, 4])], {})
