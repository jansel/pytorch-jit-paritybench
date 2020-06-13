import sys
_module = sys.modules[__name__]
del sys
generate = _module
_trackers = _module
loader = _module
dataloader = _module
loss = _module
cross_entropy = _module
models = _module
aggregator = _module
beam_onmt = _module
beam_search = _module
cond_decoder = _module
conv1d = _module
conv2d = _module
dense_modules = _module
densenet = _module
efficient_densenet = _module
embedding = _module
encoder = _module
evaluate = _module
gnmt = _module
log_efficient_densenet = _module
lstm = _module
modules = _module
norm = _module
penalties_onmt = _module
pervasive = _module
pooling = _module
seq2seq = _module
setup = _module
transitions = _module
optimizer = _module
params = _module
gpu = _module
parse = _module
utils = _module
trainer = _module
logging = _module
preprocess = _module
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


import logging


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch as t


from math import sqrt


import math


import torch.utils.checkpoint as cp


from collections import OrderedDict


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from math import log2


from math import floor


from torch import Tensor as T


import itertools


def to_contiguous(tensor):
    """
    Return a contiguous tensor
    Especially after: narrow() , view() , expand() or transpose()
    """
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def get_smooth_ml_loss(logp, target, mask, norm=True, eps=0):
    """
    Cross entropy with label smoothing
    """
    seq_length = logp.size(1)
    target = target[:, :seq_length]
    mask = mask[:, :seq_length]
    binary_mask = mask
    logp = to_contiguous(logp).view(-1, logp.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    ml_output = -logp.gather(1, target) * mask
    ml_output = torch.sum(ml_output)
    smooth_loss = -logp.sum(dim=1, keepdim=True) * mask
    smooth_loss = smooth_loss.sum() / logp.size(1)
    if norm:
        ml_output /= torch.sum(binary_mask)
        smooth_loss /= torch.sum(binary_mask)
    output = (1 - eps) * ml_output + eps * smooth_loss
    return output, ml_output


class SmoothMLCriterion(nn.Module):
    """
    Label smoothed cross entropy loss
    """

    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = params.get('mask_threshold', 1)
        self.normalize_batch = bool(params.get('normalize_batch', 1))
        self.eps = params.get('label_smoothing', 0.1)
        self.version = 'label smoothed ml'

    def log(self):
        self.logger.info('Label smoothed ML loss with eps=%.2e' % self.eps)

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        mask = target.gt(self.th_mask).float()
        output, ml_loss = get_smooth_ml_loss(logp, target, mask, norm=self.
            normalize_batch, eps=self.eps)
        return {'final': output, 'ml': ml_loss}, {}


class MLCriterion(nn.Module):
    """
    The default cross entropy loss
    """

    def __init__(self, job_name, params):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.th_mask = params.get('mask_threshold', 1)
        self.normalize = params.get('normalize', 'ntokens')
        self.version = 'ml'

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        output = self.get_ml_loss(logp, target)
        return {'final': output, 'ml': output}, {}

    def get_ml_loss(self, logp, target):
        """
        Compute the usual ML loss
        """
        batch_size = logp.size(0)
        seq_length = logp.size(1)
        vocab = logp.size(2)
        target = target[:, :seq_length]
        logp = to_contiguous(logp).view(-1, logp.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = target.gt(self.th_mask)
        ml_output = -logp.gather(1, target)[mask]
        ml_output = torch.sum(ml_output)
        if self.normalize == 'ntokens':
            norm = torch.sum(mask)
            ml_output /= norm.float()
        elif self.normalize == 'seqlen':
            norm = seq_length
            ml_output /= norm
        elif self.normalize == 'batch':
            norm = batch_size
            ml_output /= norm
        else:
            raise ValueError('Unknown normalizing scheme')
        return ml_output


class MLCriterionNLL(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """

    def __init__(self, job_name, params, pad_token):
        super().__init__()
        self.logger = logging.getLogger(job_name)
        self.pad_token = pad_token
        self.normalize_batch = params['normalize_batch']
        self.penalize_confidence = params['penalize_confidence']
        self.sentence_avg = False
        self.version = 'ml'

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target, ntokens):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        """
        logp = logp.view(-1, logp.size(-1))
        loss = F.nll_loss(logp, target.view(-1), size_average=False,
            ignore_index=self.pad_token, reduce=True)
        None
        sample_size = target.size(0) if self.sentence_avg else ntokens
        None
        output = loss / sample_size
        None
        return {'final': output, 'ml': output}, {}

    def track(self, logp, target, mask, add_dirac=False):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        """
        N = logp.size(0)
        seq_length = logp.size(1)
        target = target[:, :seq_length].data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        target_d = np.zeros_like(logp)
        rows = np.arange(N).reshape(-1, 1).repeat(seq_length, axis=1)
        cols = np.arange(seq_length).reshape(1, -1).repeat(N, axis=0)
        target_d[rows, cols, target] = 1
        return logp, target_d


def average_code(tensor, *args):
    return tensor.mean(dim=3)


def truncated_mean(tensor, src_lengths, *args):
    """
    Average-pooling up to effective legth
    """
    Pool = []
    Attention = []
    for n in range(tensor.size(0)):
        X = tensor[n]
        xpool = X[:, :, :src_lengths[n]].mean(dim=2)
        xpool *= math.sqrt(src_lengths[n])
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    return result


def truncated_max(tensor, src_lengths, track=False, *args):
    """
    Max-pooling up to effective legth
    """
    Pool = []
    Attention = []
    for n in range(tensor.size(0)):
        X = tensor[n]
        xpool, attn = X[:, :, :src_lengths[n]].max(dim=2)
        if track:
            targets = torch.arange(src_lengths[n])
            align = targets.apply_(lambda k: sum(attn[:, (-1)] == k))
            align = align / align.sum()
            Attention.append(align.unsqueeze(0))
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    if track:
        return result, torch.cat(Attention, dim=0).cuda()
    return result


def max_code(tensor, src_lengths=None, track=False):
    if track:
        batch_size, nchannels, _, max_len = tensor.size()
        xpool, attn = tensor.max(dim=3)
        targets = torch.arange(max_len).type_as(attn)
        align = []
        activ_distrib = []
        activ = []
        for n in range(batch_size):
            align.append(np.array([(torch.sum(attn[(n), :, (-1)] == k, dim=
                -1).data.item() / nchannels) for k in targets]))
            activ_distrib.append(np.array([torch.sum((attn[(n), :, (-1)] ==
                k).float() * xpool[(n), :, (-1)], dim=-1).data.item() for k in
                targets]))
            activ.append(np.array([((attn[(n), :, (-1)] == k).float() *
                xpool[(n), :, (-1)]).data.cpu().numpy() for k in targets]))
        align = np.array(align)
        activ = np.array(activ)
        activ_distrib = np.array(activ_distrib)
        return xpool, (None, align, activ_distrib, activ)
    else:
        return tensor.max(dim=3)[0]


class Aggregator(nn.Module):

    def __init__(self, input_channls, force_output_channels=None, params={}):
        nn.Module.__init__(self)
        mode = params.get('mode', 'max')
        mapping = params.get('mapping', 'linear')
        num_fc = params.get('num_fc', 1)
        self.output_channels = input_channls
        if mode == 'mean':
            self.project = average_code
        elif mode == 'max':
            self.project = max_code
        elif mode == 'truncated-max':
            self.project = truncated_max
        elif mode == 'truncated-mean':
            self.project = truncated_mean
        elif mode == 'max-attention':
            self.project = MaxAttention(params, input_channls)
            self.output_channels *= 2 - (params['first_aggregator'] == 'skip')
        else:
            raise ValueError('Unknown mode %s' % mode)
        self.add_lin = 0
        None
        if force_output_channels is not None:
            self.add_lin = 1
            assert self.output_channels > force_output_channels, 'Avoid decompressing the channels'
            None
            if num_fc == 1:
                lin = nn.Linear(self.output_channels, force_output_channels)
                None
            elif num_fc == 2:
                interm = (self.output_channels + force_output_channels) // 2
                lin = nn.Sequential(nn.Linear(self.output_channels, interm),
                    nn.ReLU(inplace=True), nn.Linear(interm,
                    force_output_channels))
                None
            else:
                raise ValueError('Not yet implemented')
            if mapping == 'linear':
                self.lin = lin
            elif mapping == 'tanh':
                self.lin = nn.Sequential(lin, nn.Tanh())
            elif mapping == 'relu':
                self.lin = nn.Sequential(lin, nn.ReLU(inplace=True))
            self.output_channels = force_output_channels

    def forward(self, tensor, src_lengths, track=False, *args):
        if not track:
            proj = self.project(tensor, src_lengths, track, *args)
            proj = proj.permute(0, 2, 1)
            if self.add_lin:
                return self.lin(proj)
            else:
                return proj
        else:
            proj, attn = self.project(tensor, src_lengths, track, *args)
            proj = proj.permute(0, 2, 1)
            if self.add_lin:
                proj = self.lin(proj)
            return proj, attn


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos, n_best=1, cuda=False,
        global_scorer=None, min_length=0, stepwise_penalty=False,
        block_ngram_repeat=0, exclusion_tokens=set()):
        self.size = size
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [self.tt.LongTensor(size).fill_(pad)]
        self.next_ys[0][0] = bos
        self._eos = eos
        self.eos_top = False
        self.attn = []
        self.finished = []
        self.n_best = n_best
        self.global_scorer = global_scorer
        self.global_state = {}
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e+20
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(
                word_probs)
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e+20
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -1e+21
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, 
            True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)
        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True
        return self.done()

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1
        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class MaskedConv1d(nn.Conv1d):
    """
    Masked (autoregressive) conv1d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
        dilation=1, groups=1, bias=False):
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
            kernel_size, padding=padding, groups=groups, dilation=dilation,
            bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        self.incremental_state = t.zeros(1, 1, 1)
        _, _, kH = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

    def update(self, x):
        k = self.kernel_size // 2 + 1
        buffer = self.incremental_state
        if buffer.size(-1) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            buffer[:, :, :-1] = buffer[:, :, 1:].clone()
            buffer[:, :, -1:] = x[:, :, -1:]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class AsymmetricMaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d kx1 kernel
    FIXME: particular case of the MaskedConv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
        groups=1, bias=False):
        pad = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, (kernel_size, 1),
            padding=(pad, 0), groups=groups, dilation=dilation, bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super().forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class MaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
        groups=1, bias=False):
        pad = dilation * (kernel_size - 1) // 2
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
            kernel_size, padding=pad, groups=groups, dilation=dilation,
            bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class _MainDenseLayer(nn.Module):
    """
    Main dense layer declined in 2 variants
    """

    def __init__(self, num_input_features, kernel_size, params):
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_size = params.get('bn_size', 4)
        self.growth_rate = params.get('growth_rate', 32)
        self.drop_rate = params.get('conv_dropout', 0.0)

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


class DenseLayer_Asym(nn.Module):
    """
    Dense layer with asymmetric convolution ie decompose a 3x3 conv into
    a 3x1 1D conv followed by a 1x3 1D conv.
    As suggested in: 
    Efficient Dense Modules of Asymmetric Convolution for
    Real-Time Semantic Segmentation
    https://arxiv.org/abs/1809.06323
    """

    def __init__(self, num_input_features, kernel_size, params, first=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.drop_rate = params.get('conv_dropout', 0.0)
        bias = params.get('bias', 0)
        bn_size = params.get('bn_size', 4)
        growth_rate = params.get('growth_rate', 32)
        dim1 = bn_size * growth_rate
        dim2 = bn_size // 2 * growth_rate
        conv1 = nn.Conv2d(num_input_features, dim1, kernel_size=1, bias=False)
        pad = (kernel_size - 1) // 2
        conv2s = nn.Conv2d(dim1, dim2, kernel_size=(1, kernel_size),
            padding=(0, pad), bias=False)
        conv2t = AsymmetricMaskedConv2d(dim2, growth_rate, kernel_size=
            kernel_size, bias=False)
        self.seq = nn.Sequential(nn.BatchNorm2d(num_input_features), nn.
            ReLU(inplace=True), conv1, nn.BatchNorm2d(dim1), nn.ReLU(
            inplace=True), conv2s, conv2t)

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


class GatedConv2d(MaskedConv2d):
    """
    Gated version of the masked conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
        bias=False, groups=1):
        super(GatedConv2d, self).__init__(in_channels, 2 * out_channels,
            kernel_size, dilation=dilation, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super(GatedConv2d, self).forward(x)
        mask, out = x.chunk(2, dim=1)
        mask = self.sigmoid(mask)
        return out * mask


def _setup_conv_dilated(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.0)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    dilation = params.get('dilation', 2)
    print('Dilation: ', dilation)
    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(num_input_features, interm_features, kernel_size=1,
        bias=bias)
    conv2 = CV(interm_features, interm_features, kernel_size=kernel_size,
        bias=bias)
    conv3 = CV(interm_features, growth_rate, kernel_size=kernel_size, bias=
        bias, dilation=dilation)
    if init_weights == 'manual':
        if not first:
            cst = 2 * (1 - drop_rate)
        else:
            cst = 1
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        std2 = sqrt(2 / (interm_featires * kernel_size * (kernel_size - 1) //
            2))
        conv2.weight.data.normal_(0, std2)
        conv3.weight.data.normal_(0, std2)
        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()
            conv3.bias.data.zero_()
    elif init_weights == 'kaiming':
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity=
            'relu')
        nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity=
            'relu')
        nn.init.kaiming_normal_(conv3.weight, mode='fan_out', nonlinearity=
            'relu')
    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0)
        conv2 = nn.utils.weight_norm(conv2, dim=0)
        conv3 = nn.utils.weight_norm(conv3, dim=0)
    return conv1, conv2, conv3


class DenseLayer_Dil(_MainDenseLayer):
    """
    BN > ReLU > Conv(1)
    > BN > ReLU > Conv(k)
    > BN > ReLU > Conv(k, dilated)

    """

    def __init__(self, num_input_features, kernel_size, params, first=False):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2, conv3 = _setup_conv_dilated(num_input_features,
            kernel_size, params)
        self.seq = nn.Sequential(nn.BatchNorm2d(num_input_features), nn.
            ReLU(inplace=True), conv1, nn.BatchNorm2d(self.bn_size * self.
            growth_rate), nn.ReLU(inplace=True), conv2, nn.BatchNorm2d(self
            .bn_size * self.growth_rate), nn.ReLU(inplace=True), conv3)


def _setup_conv(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.0)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    depthwise = params.get('depthwise', 0)
    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(num_input_features, interm_features, kernel_size=1,
        bias=bias)
    gp = growth_rate if depthwise else 1
    conv2 = CV(interm_features, growth_rate, kernel_size=kernel_size, bias=
        bias, groups=gp)
    if init_weights == 'manual':
        if not first:
            cst = 2 * (1 - drop_rate)
        else:
            cst = 1
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        std2 = sqrt(2 / (bn_size * growth_rate * kernel_size * (kernel_size -
            1) // 2))
        conv2.weight.data.normal_(0, std2)
        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()
    elif init_weights == 'kaiming':
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity=
            'relu')
        nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity=
            'relu')
    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0)
        conv2 = nn.utils.weight_norm(conv2, dim=0)
    return conv1, conv2


class DenseLayer(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > BN > ReLU > Conv(k)
    """

    def __init__(self, num_input_features, kernel_size, params, first=False):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(nn.BatchNorm2d(num_input_features), nn.
            ReLU(inplace=True), conv1, nn.BatchNorm2d(self.bn_size * self.
            growth_rate), nn.ReLU(inplace=True), conv2)


class DenseLayer_midDP(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > Dropout > BN > ReLU > Conv(k)
    """

    def __init__(self, num_input_features, kernel_size, params, first=False):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(nn.BatchNorm2d(num_input_features), nn.
            ReLU(inplace=True), conv1, nn.Dropout(p=self.drop_rate, inplace
            =True), nn.BatchNorm2d(self.bn_size * self.growth_rate), nn.
            ReLU(inplace=True), conv2)


class DenseLayer_noBN(_MainDenseLayer):
    """
    ReLU > Conv(1) > ReLU > Conv(k)
    #TODO: check activ' var
    """

    def __init__(self, num_input_features, kernel_size, params, first=False):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params,
            first=first)
        self.seq = nn.Sequential(nn.ReLU(inplace=True), conv1, nn.ReLU(
            inplace=True), conv2)


class DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, kernels, params):
        super(DenseBlock, self).__init__()
        layer_type = params.get('layer_type', 1)
        growth_rate = params.get('growth_rate', 32)
        if layer_type == 'regular':
            LayerModule = DenseLayer
        elif layer_type == 'mid-dropout':
            LayerModule = DenseLayer_midDP
        elif layer_type == 'nobn':
            LayerModule = DenseLayer_noBN
        elif layer_type == 'asym':
            LayerModule = DenseLayer_Asym
        elif layer_type == 'dilated':
            LayerModule = DenseLayer_Dil
        else:
            raise ValueError('Unknown type: %d' % layer_type)
        None
        for i in range(num_layers):
            None
            layer = LayerModule(num_input_features + i * growth_rate,
                kernels[i], params, first=i == 0)
            self.add_module('denselayer%d' % (i + 1), layer)

    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()

    def track(self, x):
        activations = []
        for layer in list(self.children()):
            x, newf = layer.track(x)
            activations.append(newf.data.cpu().numpy())
            x = torch.cat([x, newf], 1)
        return x, activations


class DenseNet(nn.Module):

    def __init__(self, num_init_features, params):
        super(DenseNet, self).__init__()
        block_layers = params.get('num_layers', 24)
        block_kernels = params['kernels']
        growth_rate = params.get('growth_rate', 32)
        divide_channels = params.get('divide_channels', 2)
        init_weights = params.get('init_weights', 0)
        normalize_channels = params.get('normalize_channels', 0)
        transition_type = params.get('transition_type', 1)
        skip_last_trans = params.get('skip_last_trans', 0)
        if transition_type == 1:
            TransitionLayer = Transition
        elif transition_type == 2:
            TransitionLayer = Transition2
        self.features = nn.Sequential()
        num_features = num_init_features
        if normalize_channels:
            self.features.add_module('initial_norm', nn.GroupNorm(1,
                num_features))
        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == 'manual':
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels
        for i, (num_layers, kernels) in enumerate(zip(block_layers,
            block_kernels)):
            block = DenseBlock(num_layers, num_features, kernels, params)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if not i == len(block_layers) - 1 or not skip_last_trans:
                trans = TransitionLayer(num_input_features=num_features,
                    num_output_features=num_features // 2, init_weights=
                    init_weights)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                None
        None
        self.output_channels = num_features
        self.features.add_module('norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                layer.reset_buffers()

    def track(self, x):
        activations = []
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x, actv = layer.track(x)
                activations.append(actv)
            else:
                x = layer(x)
        return x, activations


def _bn_function_factory(norm, relu, conv, index, mode=1):
    if mode == 1:
        connexions = [(index - 2 ** k) for k in range(1 + floor(log2(index)))]
    elif mode == 2:
        connexions = [(index - 2 ** k) for k in range(1 + floor(log2(index)))]
        if 0 not in connexions:
            connexions.append(0)

    def bn_function(*inputs):
        concatenated_features = torch.cat([inputs[c] for c in connexions], 1)
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, kernel_size=3,
        bn_size=4, drop_rate=0, gated=False, bias=False, init_weights=0,
        weight_norm=False, efficient=False):
        super(_DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        if gated:
            CV = GatedConv2d
        else:
            CV = MaskedConv2d
        conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
            kernel_size=1, bias=bias)
        conv2 = CV(bn_size * growth_rate, growth_rate, kernel_size=
            kernel_size, bias=bias)
        if init_weights == 'manual':
            std1 = sqrt(2 / num_input_features)
            conv1.weight.data.normal_(0, std1)
            std2 = sqrt(2 * (1 - drop_rate) / (bn_size * growth_rate *
                kernel_size * (kernel_size - 1) // 2))
            conv2.weight.data.normal_(0, std2)
            if bias:
                conv1.bias.data.zero_()
                conv2.bias.data.zero_()
        elif init_weights == 'kaiming':
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out',
                nonlinearity='relu')
            nn.init.kaiming_normal_(conv2.weight, mode='fan_out',
                nonlinearity='relu')
        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1, dim=0)
            conv2 = nn.utils.weight_norm(conv2, dim=0)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', conv1)
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', conv2)

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for
            prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return new_features

    def reset_buffers(self):
        self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2.update(self.relu2(self.norm2(x)))
        return torch.cat([res, x], 1)


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, num_input_features, kernels, bn_size,
        growth_rate, drop_rate, gated, bias, init_weights, weight_norm,
        efficient=False):
        super(_DenseBlock, self).__init__()
        None
        for i in range(num_layers):
            None
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, kernels[i], bn_size, drop_rate, gated=gated,
                bias=bias, init_weights=init_weights, weight_norm=
                weight_norm, efficient=efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()


class Efficient_DenseNet(nn.Module):
    """ 
    efficient (bool):
    set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, num_init_features, params):
        super(Efficient_DenseNet, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        block_layers = params.get('num_layers', (6, 12, 24, 16))
        block_kernels = params['kernels']
        bn_size = params.get('bn_size', 4)
        drop_rate = params.get('conv_dropout', 0)
        gated = params.get('gated', 0)
        bias = bool(params.get('bias', 1))
        init_weights = params.get('init_weights', 0)
        weight_norm = params.get('weight_norm', 0)
        divide_channels = params.get('divide_channels', 2)
        efficient = params.get('efficient', 0)
        self.features = nn.Sequential()
        num_features = num_init_features
        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == 'manual':
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels
        for i, (num_layers, kernels) in enumerate(zip(block_layers,
            block_kernels)):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, kernels=kernels, bn_size=bn_size, growth_rate
                =growth_rate, drop_rate=drop_rate, gated=gated, bias=bias,
                init_weights=init_weights, weight_norm=weight_norm,
                efficient=efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = Transition(num_input_features=num_features,
                num_output_features=num_features // 2, init_weights=
                init_weights)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
            None
        self.output_channels = num_features
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, _DenseBlock):
                layer.reset_buffers()


def make_positions(tensor, padding_idx, left_pad):
    len = tensor.size(1)
    max_pos = padding_idx + 1 + len
    out = torch.arange(padding_idx + 1, max_pos).long().cuda()
    mask = tensor.ne(padding_idx)
    positions = out[:len].expand_as(tensor)
    final = tensor.clone().masked_scatter_(mask, positions[mask])
    if left_pad:
        zero_left = torch.zeros(tensor.size(0), 1).type_as(final)
        final = torch.cat([torch.zeros(tensor.size(0), 1).type_as(final),
            final[:, :-1]], dim=1)
    return final


class PosEmbedding(nn.Embedding):

    def __init__(self, max_length, position_dim, pad_left=False):
        super(PosEmbedding, self).__init__(max_length, position_dim, 0)
        self.pad_left = pad_left

    def forward(self, labels):
        positions = make_positions(labels, self.padding_idx, self.pad_left)
        return super().forward(positions)

    def map(self, inputs):
        return super(PosEmbedding, self).forward(inputs)


class Embedding(nn.Module):

    def __init__(self, params, vocab_size, padding_idx, pad_left=False):
        nn.Module.__init__(self)
        self.dimension = params['input_dim']
        self.encode_length = params['encode_length']
        self.encode_position = params['encode_position']
        self.dropout = params['input_dropout']
        self.init_std = params.get('init_std', 0.01)
        self.zero_pad = params.get('zero_pad', 0)
        self.padding_idx = padding_idx
        self.label_embedding = nn.Embedding(vocab_size, self.dimension,
            padding_idx, scale_grad_by_freq=False)
        if self.encode_position:
            self.pos_embedding = PosEmbedding(params['max_length'], self.
                dimension, pad_left=pad_left)
        if self.encode_length:
            self.dimension += self.encode_length
            self.length_embedding = nn.Embedding(params['max_length'], self
                .encode_length)

    def init_weights(self):
        std = self.init_std
        self.label_embedding.weight.data.normal_(0, std)
        if self.zero_pad:
            self.label_embedding.weight.data[self.padding_idx].fill_(0)
        if self.encode_position:
            self.pos_embedding.weight.data.normal_(0, std)
        if self.encode_length:
            self.length_embedding.weight.data.normal_(0, std)

    def forward(self, data):
        labels = data['labels']
        emb = self.label_embedding(labels)
        if self.encode_position:
            pos = self.pos_embedding(labels)
            emb = sqrt(0.5) * (emb + pos)
        if self.encode_length:
            lens = self.length_embedding(data['lengths']).unsqueeze(1).repeat(
                1, emb.size(1), 1)
            emb = torch.cat((emb, lens), dim=2)
        if self.dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        return emb

    def single_token(self, tok, position, length=None):
        emb = self.label_embedding(tok)
        if self.encode_position:
            position = torch.ones((tok.size(0), 1)).type_as(tok) * position
            pos = self.pos_embedding.map(position)
            emb += pos
        if self.encode_length:
            lens = self.length_embedding(length).unsqueeze(1).repeat(1, emb
                .size(1), 1)
            emb = torch.cat((emb, lens), dim=2)
        if self.dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        return emb

    def reset_buffers(self):
        pass


def read_list(param):
    """ Parse list of integers """
    return [int(p) for p in str(param).split(',')]


class ConvEmbedding(nn.Module):
    """
    A 1d convolutional network on top of the lookup embeddings.
    TODO : à la ELMO, learnable combination of the layers activations
    TODO : add skip connections
    """

    def __init__(self, params, vocab_size, padding_idx, is_target=False):
        nn.Module.__init__(self)
        self.dimension = params['input_dim']
        self.encode_length = params['encode_length']
        self.encode_position = params['encode_position']
        self.dropout = params['input_dropout']
        self.init_std = params.get('init_std', 0.01)
        self.nlayers = params['num_layers']
        kernels = read_list(params['kernels'])
        out_channels = read_list(params['channels'])
        assert len(out_channels
            ) == self.nlayers, 'Number of channels should match the depth'
        assert len(kernels
            ) == self.nlayers, 'Number of kernel sizes should match the depth'
        out_channels.insert(0, self.dimension)
        None
        self.kernel_size = max(kernels)
        self.label_embedding = nn.Embedding(vocab_size, self.dimension,
            padding_idx, scale_grad_by_freq=False)
        self.conv = nn.Sequential()
        if is_target:
            conv = MaskedConv1d
            self.incremental_state = None
        else:
            conv = nn.Conv1d
        for l in range(self.nlayers):
            kernel = kernels[l]
            pad = (kernel - 1) // 2
            self.conv.add_module('conv%d' % l, conv(out_channels[l],
                out_channels[l + 1], kernel, padding=pad, bias=False))
            None
        self.dimension = out_channels[-1]

    def init_weights(self):
        self.label_embedding.weight.data.normal_(0, self.init_std)

    def forward(self, data):
        labels = data['labels']
        emb = self.label_embedding(labels)
        emb = emb.permute(0, 2, 1)
        emb = self.conv(emb)
        if self.dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = emb.permute(0, 2, 1)
        return emb

    def single_token(self, labels, position=0):
        if self.incremental_state is not None:
            if self.incremental_state.size(1) >= self.kernel_size:
                buffer = self.incremental_state
                buffer[:, :-1] = buffer[:, 1:].clone()
                buffer[:, -1:] = labels[:, -1:]
                labels = buffer
            else:
                buffer = self.incremental_state
                buffer = torch.cat((buffer, labels), dim=1)
                labels = buffer
        self.incremental_state = labels
        emb = self.label_embedding(labels)
        emb = emb.permute(0, 2, 1)
        for cvl in list(self.conv.children()):
            emb = cvl(emb)
        emb = emb.permute(0, 2, 1)
        return emb

    def reset_buffers(self):
        self.incremental_state = None
        for clv in list(self.conv.children()):
            clv.incremental_state = torch.zeros(1, 1, 1)


class Encoder(nn.Module):

    def __init__(self, params, vocab_size):
        nn.Module.__init__(self)
        self.input_dim = params['input_dim']
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.bidirectional = params['bidirectional']
        self.nd = 2 if self.bidirectional else 1
        self.cell_type = params['cell_type'].upper()
        self.nlayers = params['num_layers']
        self.size = params['cell_dim']
        self.parallel = params['parallel']
        self.hidden_dim = self.size // self.nd
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim, self
            .pad_token, scale_grad_by_freq=bool(params['scale_grad_by_freq']))
        self.input_dropout = nn.Dropout(params['input_dropout'])
        if params['cell_dropout'] and self.nlayers == 1:
            params['cell_dropout'] = 0
        self.cell = getattr(nn, self.cell_type)(self.input_dim, self.
            hidden_dim, self.nlayers, bidirectional=self.bidirectional,
            batch_first=True, dropout=params['cell_dropout'])

    def init_weights(self):
        """Initialize weights."""
        initdev = 0.01
        self.embedding.weight.data.normal_(0.0, initdev)
        self.embedding.weight.data.normal_(0.0, initdev)

    def init_state(self, batch_size):
        """Get cell states and hidden states."""
        h0 = torch.zeros(self.nlayers * self.nd, batch_size, self.hidden_dim)
        if self.cell_type == 'GRU':
            return h0
        c0 = torch.zeros(self.nlayers * self.nd, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, data):
        labels = data['labels']
        lengths = data['lengths']
        batch_size = labels.size(0)
        emb = self.input_dropout(self.embedding(labels))
        _emb = emb
        pack_emb = pack_padded_sequence(emb, lengths, batch_first=True)
        state = self.init_state(batch_size)
        ctx, state = self.cell(pack_emb, state)
        ctx, _ = pad_packed_sequence(ctx, batch_first=True)
        if self.bidirectional:
            if self.cell_type == 'LSTM':
                h_t = torch.cat((state[0][-1], state[0][-2]), 1)
                c_t = torch.cat((state[1][-1], state[1][-2]), 1)
                final_state = [h_t, c_t]
            elif self.cell_type == 'GRU':
                h_t = torch.cat((state[-1], state[-2]), 1)
                final_state = [h_t]
        elif self.cell_type == 'LSTM':
            h_t = state[0][-1]
            c_t = state[1][-1]
            final_state = [h_t, c_t]
        elif self.cell_type == 'GRU':
            h_t = state[0][-1]
            final_state = [h_t]
        return {'emb': _emb, 'ctx': ctx, 'state': final_state}


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, kernel_size, params, index):
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_size = params.get('bn_size', 4)
        self.growth_rate = params.get('growth_rate', 32)
        self.drop_rate = params.get('conv_dropout', 0.0)
        self.index = index
        self.mode = params.get('log_mode', 1)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', conv1)
        self.add_module('norm2', nn.BatchNorm2d(self.bn_size * self.
            growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', conv2)

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.
            conv1, self.index, self.mode)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return new_features


def is_power2(num):
    """ True iff integer is a power of 2"""
    return num & num - 1 == 0 and num != 0


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, num_input_features, kernels, params):
        super(_DenseBlock, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        log_mode = params.get('log_mode', 1)
        None
        for i in range(num_layers):
            index = i + 1
            numc = floor(log2(index)) + 1
            if log_mode == 1:
                if is_power2(index):
                    numf = num_input_features + (numc - 1) * growth_rate
                else:
                    numf = numc * growth_rate
            elif log_mode == 2:
                if is_power2(index):
                    numf = num_input_features + (numc - 1) * growth_rate
                else:
                    numf = numc * growth_rate + num_input_features
            None
            layer = _DenseLayer(numf, kernels[i], params, i + 1)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class Log_Efficient_DenseNet(nn.Module):
    """ 
    set to True to use checkpointing. Much more memory efficient, but slower.
    log connections inside a block:
    x_i = f_i(concat({x_{i-[2^k]}, i < [log(i)]}))
    Implementation of Log-Densenet V1 described in:
    ``LOG-DENSENET: HOW TO SPARSIFY A DENSENET``
    arxiv: https://arxiv.org/pdf/1711.00002.pdf
    """

    def __init__(self, num_init_features, params):
        super(Log_Efficient_DenseNet, self).__init__()
        growth_rate = params.get('growth_rate', 32)
        block_layers = params.get('num_layers', (6, 12, 24, 16))
        block_kernels = params['kernels']
        init_weights = params.get('init_weights', 0)
        divide_channels = params.get('divide_channels', 2)
        skip_last_trans = params.get('skip_last_trans', 0)
        self.features = nn.Sequential()
        num_features = num_init_features
        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == 'manual':
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels
        for i, (num_layers, kernels) in enumerate(zip(block_layers,
            block_kernels)):
            block = _DenseBlock(num_layers, num_features, kernels, params)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if not i == len(block_layers) - 1 or not skip_last_trans:
                trans = Transition(num_input_features=num_features,
                    num_output_features=num_features // 2, init_weights=
                    init_weights)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                None
        None
        self.output_channels = num_features
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())


class AllamanisConvAttention(nn.Module):
    """
    Convolutional attention
    @inproceedings{allamanis2016convolutional,
          title={A Convolutional Attention Network for
                 Extreme Summarization of Source Code},
          author={Allamanis, Miltiadis and Peng, Hao and Sutton, Charles},
          booktitle={International Conference on Machine Learning (ICML)},
          year={2016}
      }
    """

    def __init__(self, params, enc_params):
        super(AllamanisConvAttention, self).__init__()
        src_emb_dim = enc_params['input_dim']
        dims = params['attention_channels'].split(',')
        dim1, dim2 = [int(d) for d in dims]
        None
        widths = params['attention_windows'].split(',')
        w1, w2, w3 = [int(w) for w in widths]
        None
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        self.conv1 = nn.Conv1d(src_emb_dim, dim1, w1, padding=(w1 - 1) // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, trg_dim, w2, padding=(w2 - 1) // 2)
        self.conv3 = nn.Conv1d(trg_dim, 1, w3, padding=(w3 - 1) // 2)
        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_emb_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def score(self, input, context, src_emb):
        """
        input: batch x trg_dim
        context & src_emb : batch x Tx x src_dim (resp. src_emb_dim)
        return the alphas for comuting the weighted context
        """
        src_emb = src_emb.transpose(1, 2)
        L1 = self.relu(self.conv1(src_emb))
        L2 = self.conv2(L1)
        L2 = L2 * input.unsqueeze(2).repeat(1, 1, L2.size(2))
        if self.normalize:
            norm = L2.norm(p=2, dim=1, keepdim=True)
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                None
        attn = self.conv3(L2)
        attn_sm = self.sm(attn)
        return attn_sm

    def forward(self, input, context, src_emb):
        """
        Score the context (resp src embedding)
        and return a new context as a combination of either
        the source embeddings or the hidden source codes
        """
        attn_sm = self.score(input, context, src_emb)
        weighted_context = torch.bmm(attn_sm, src_emb).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn_sm


class ConvAttentionHid(nn.Module):
    """
    Convolutional attention
    All around similar to Allamanis attention while never
    using the source word embeddings
    """

    def __init__(self, params, enc_params):
        super(ConvAttentionHid, self).__init__()
        src_dim = enc_params['cell_dim']
        dims = params['attention_channels'].split(',')
        dim1, dim2 = [int(d) for d in dims]
        None
        widths = params['attention_windows'].split(',')
        w1, w2, w3 = [int(w) for w in widths]
        None
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        self.conv1 = nn.Conv1d(src_dim, dim1, w1, padding=(w1 - 1) // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, trg_dim, w2, padding=(w2 - 1) // 2)
        self.conv3 = nn.Conv1d(trg_dim, 1, w3, padding=(w3 - 1) // 2)
        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input, context, src_emb):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        context = context.transpose(1, 2)
        L1 = self.relu(self.conv1(context))
        L2 = self.conv2(L1)
        L2 = L2 * input.unsqueeze(2).repeat(1, 1, L2.size(2))
        if self.normalize:
            norm = L2.norm(p=2, dim=2, keepdim=True)
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                None
        attn = self.conv3(L2)
        attn_sm = self.sm(attn)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context, attn_reshape).squeeze(2)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ConvAttentionHidCat(nn.Module):
    """
    Convolutional attention
    Use the encoder hidden states all around, Jakob's idea
    """

    def __init__(self, params, enc_params):
        super(ConvAttentionHidCat, self).__init__()
        src_dim = enc_params['cell_dim']
        trg_dim = params['cell_dim']
        self.normalize = params['normalize_attention']
        widths = params['attention_windows'].split(',')
        self.num_conv_layers = len(widths)
        dims = params['attention_channels'].split(',')
        assert len(dims) == self.num_conv_layers - 1
        if self.num_conv_layers == 3:
            w1, w2, w3 = [int(w) for w in widths]
            None
            dim1, dim2 = [int(d) for d in dims]
            None
        elif self.num_conv_layers == 4:
            w1, w2, w3, w4 = [int(w) for w in widths]
            None
            dim1, dim2, dim3 = [int(d) for d in dims]
            None
        else:
            raise ValueError(
                'Number of layers is either 3 or 4, still working on a general form'
                )
        self.conv1 = nn.Conv1d(src_dim + trg_dim, dim1, w1, padding=(w1 - 1
            ) // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim1, dim2, w2, padding=(w2 - 1) // 2)
        if self.num_conv_layers == 3:
            self.conv3 = nn.Conv1d(dim2, 1, w3, padding=(w3 - 1) // 2)
        elif self.num_conv_layers == 4:
            self.conv3 = nn.Conv1d(dim2, dim3, w3, padding=(w3 - 1) // 2)
            self.conv4 = nn.Conv1d(dim3, 1, w4, padding=(w4 - 1) // 2)
        self.sm = nn.Softmax(dim=2)
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input, context, src_emb):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        context = context.transpose(1, 2)
        input_cat = torch.cat((context, input.unsqueeze(2).repeat(1, 1,
            context.size(2))), 1)
        L1 = self.relu(self.conv1(input_cat))
        L2 = self.conv2(L1)
        if self.normalize:
            norm = L2.norm(p=2, dim=2, keepdim=True)
            L2 = L2.div(norm)
            if len((norm == 0).nonzero()):
                None
        if self.num_conv_layers == 3:
            attn = self.conv3(L2)
        else:
            attn = self.conv4(self.conv3(L2))
        attn_sm = self.sm(attn)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context, attn_reshape).squeeze(2)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LocalDotAttention(nn.Module):
    """
    Soft Dot/ local-predictive attention

    Ref: http://www.aclweb.org/anthology/D15-1166
    Effective approaches to attention based NMT (Luong et al. EMNLP 15)
    """

    def __init__(self, params):
        super(LocalDotAttention, self).__init__()
        dim = params['cell_dim']
        dropout = params['attention_dropout']
        self.window = 4
        self.sigma = self.window / 2
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.linear_predict_1 = nn.Linear(dim, dim // 2, bias=False)
        self.linear_predict_2 = nn.Linear(dim // 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, context, src_emb=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        Tx = context.size(1)
        pt = self.tanh(self.linear_predict_1(input))
        None
        pt = self.linear_predict_2(pt)
        None
        pt = Tx * self.sigmoid(pt)
        bl, bh = (pt - self.window).int(), (pt + self.window).int()
        indices = torch.cat([torch.arange(i.item(), j.item()).unsqueeze(0) for
            i, j in zip(bl, bh)], dim=0).long()
        None
        target = self.linear_in(input).unsqueeze(2)
        None
        context_window = context.gather(0, indices)
        None
        attn = torch.bmm(context_window, target).squeeze(2)
        attn = self.sm(self.dropout(attn))
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context_window).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Effective approaches to attention based NMT (Luong et al. EMNLP 15)
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, params):
        super(SoftDotAttention, self).__init__()
        dim = params['cell_dim']
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(params['attention_dropout'])

    def forward(self, input, context, src_emb=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        attn = self.sm(self.dropout(attn))
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class AllamanisConvAttentionBis(AllamanisConvAttention):
    """
    Similar to AllamanisConvAttention with the only difference at computing
    the weighted context which takes the encoder's hidden states
    instead of the source word embeddings
    """

    def __init__(self, params, enc_params):
        super(AllamanisConvAttentionBis, self).__init__(params)
        trg_dim = params['cell_dim']
        src_dim = enc_params['cell_dim']
        self.linear_out = nn.Linear(trg_dim + src_dim, trg_dim, bias=False)

    def forward(self, input, context, src_emb):
        attn_sm = self.score(input, context, src_emb)
        attn_reshape = attn_sm.transpose(1, 2)
        weighted_context = torch.bmm(context.transpose(1, 2), attn_reshape
            ).squeeze(2)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn_sm


class LSTMAttention(nn.Module):
    """
    A long short-term memory (LSTM) cell with attention.
    Use SoftDotAttention
    """

    def __init__(self, params, enc_params):
        super(LSTMAttention, self).__init__()
        self.mode = params['attention_mode']
        self.input_size = params['input_dim']
        self.hidden_size = params['cell_dim']
        self.input_weights = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.hidden_weights = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        if self.mode == 'dot':
            self.attention_layer = SoftDotAttention(params)
        elif self.mode == 'local-dot':
            self.attention_layer = LocalDotAttention(params)
        elif self.mode == 'allamanis':
            self.attention_layer = AllamanisConvAttention(params, enc_params)
        elif self.mode == 'allamanis-v2':
            self.attention_layer = AllamanisConvAttentionBis(params, enc_params
                )
        elif self.mode == 'conv-hid':
            self.attention_layer = ConvAttentionHid(params, enc_params)
        elif self.mode == 'conv-hid-cat':
            self.attention_layer = ConvAttentionHidCat(params, enc_params)
        else:
            raise ValueError('Unkown attention mode %s' % self.mode)

    def forward(self, input, hidden, ctx, src_emb):
        """Propogate input through the network."""

        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden
            gates = self.input_weights(input) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)
            cy = forgetgate * cx + ingate * cellgate
            hy = outgate * F.tanh(cy)
            h_tilde, attn = self.attention_layer(hy, ctx, src_emb)
            return (h_tilde, cy), attn
        input = input.transpose(0, 1)
        output = []
        attention = []
        steps = list(range(input.size(0)))
        for i in steps:
            hidden, attn = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                h = hidden[0]
            else:
                h = hidden
            output.append(h)
            attention.append(attn)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        attention = torch.cat(attention, 0)
        return output, hidden, attention


class LSTM(nn.LSTM):

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)

    def forward(self, input, hidden, ctx, src_emb):
        if hidden[0].size(0) != 1:
            hidden = [h.unsqueeze(0) for h in hidden]
        output, hdec = super(LSTM, self).forward(input, hidden)
        return output, hdec


class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):
        """Sinusoidal Positional_Encoding.
        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values
                    of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied
                 by sqrt num_units(check details from paper)
        """
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        N, T = inputs.size()[0:2]
        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1
            ).long()
        position_enc = torch.Tensor([[(pos / np.power(10000, 2.0 * i / self
            .num_units)) for i in range(self.num_units)] for pos in range(T)])
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        lookup_table = position_enc
        if self.zeros_pad:
            lookup_table = torch.cat((torch.zeros(1, self.num_units),
                lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1
        outputs = F.embedding(position_ind, lookup_table, padding_idx, None,
            2, False, False)
        if self.scale:
            outputs = outputs * self.num_units ** 0.5
        return outputs


class ChannelsNormalization(nn.Module):

    def __init__(self, n_channels, eps=0.001):
        super(ChannelsNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(1, n_channels, 1, 1),
            requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(1, n_channels, 1, 1),
            requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, keepdim=True, dim=1)
        sigma = torch.std(z, keepdim=True, dim=1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(
            ln_out)
        return ln_out


class LayerNormalization(nn.Module):

    def __init__(self, d_hid, eps=0.001):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(
            ln_out)
        return ln_out


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-06):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        self.alpha = nn.Parameter(self.alpha)
        self.beta = nn.Parameter(self.beta)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).expand_as(x)) / torch.sqrt(torch.var(x, 1
            ).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


def _expand(tensor, dim, reps):
    if dim == 1:
        return tensor.repeat(1, reps, 1, 1)
    if dim == 2:
        return tensor.repeat(1, 1, reps, 1)
    else:
        raise NotImplementedError


class Pervasive(nn.Module):

    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
        special_tokens):
        nn.Module.__init__(self)
        self.logger = logging.getLogger(jobname)
        self.version = 'conv'
        self.params = params
        self.merge_mode = params['network'].get('merge_mode', 'concat')
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = special_tokens['PAD']
        self.mask_version = params.get('mask_version', -1)
        self.bos_token = special_tokens['BOS']
        self.eos_token = special_tokens['EOS']
        self.kernel_size = max(list(itertools.chain.from_iterable(params[
            'network']['kernels'])))
        if params['encoder']['type'] == 'none':
            self.src_embedding = Embedding(params['encoder'],
                src_vocab_size, padding_idx=self.padding_idx)
        elif params['encoder']['type'] == 'conv':
            self.src_embedding = ConvEmbedding(params['encoder'],
                src_vocab_size, padding_idx=self.padding_idx)
        self.trg_embedding = Embedding(params['decoder'], trg_vocab_size,
            padding_idx=self.padding_idx, pad_left=True)
        if self.merge_mode == 'concat':
            self.input_channels = (self.src_embedding.dimension + self.
                trg_embedding.dimension)
        elif self.merge_mode == 'product':
            self.input_channels = self.src_embedding.dimension
        elif self.merge_mode == 'bilinear':
            bilinear_dim = params['network'].get('bilinear_dimension', 128)
            self.input_channels = bilinear_dim
            std = params['encoder'].get('init_std', 0.01)
            self.bw = nn.Parameter(std * torch.randn(bilinear_dim))
        elif self.merge_mode == 'multi-sim':
            self.sim_dim = params['network'].get('similarity_dimension', 128)
            self.input_channels = self.sim_dim
            std = params['encoder'].get('init_std', 0.01)
            self.bw = nn.Parameter(std * torch.randn(self.sim_dim, self.
                trg_embedding.dimension, self.src_embedding.dimension))
        elif self.merge_mode == 'multi-sim2':
            self.sim_dim = params['network'].get('similarity_dimension', 128)
            self.input_channels = self.sim_dim
            std = params['encoder'].get('init_std', 0.01)
            self.bw = nn.Parameter(torch.empty(self.sim_dim, self.
                trg_embedding.dimension, self.src_embedding.dimension))
            nn.init.orthogonal_(self.bw)
        else:
            raise ValueError('Unknown merging mode')
        self.logger.info('Model input channels: %d', self.input_channels)
        self.logger.info('Selected network: %s', params['network']['type'])
        if params['network']['divide_channels'] > 1:
            self.logger.warning('Reducing the input channels by %d', params
                ['network']['divide_channels'])
        if params['network']['type'] == 'densenet':
            self.net = DenseNet(self.input_channels, params['network'])
            self.network_output_channels = self.net.output_channels
        elif params['network']['type'] == 'efficient-densenet':
            self.net = Efficient_DenseNet(self.input_channels, params[
                'network'])
            self.network_output_channels = self.net.output_channels
        elif params['network']['type'] == 'log-densenet':
            self.net = Log_Efficient_DenseNet(self.input_channels, params[
                'network'])
            self.network_output_channels = self.net.output_channels
        else:
            raise ValueError('Unknown architecture %s' % params['network'][
                'type'])
        self.tie_target_weights = params['decoder']['tie_target_weights']
        self.copy_source_weights = params['decoder']['copy_source_weights']
        if self.tie_target_weights:
            self.logger.warning('Tying the decoder weights')
            last_dim = params['decoder']['input_dim']
        else:
            last_dim = None
        self.aggregator = Aggregator(self.network_output_channels, last_dim,
            params['aggregator'])
        self.final_output_channels = self.aggregator.output_channels
        self.prediction_dropout = nn.Dropout(params['decoder'][
            'prediction_dropout'])
        self.logger.info('Output channels: %d', self.final_output_channels)
        self.prediction = nn.Linear(self.final_output_channels, self.
            trg_vocab_size)
        if self.copy_source_weights:
            self.trg_embedding.label_embedding.weight = (self.src_embedding
                .label_embedding.weight)
        if self.tie_target_weights:
            self.prediction.weight = self.trg_embedding.label_embedding.weight

    def init_weights(self):
        """
        Called after setup.buil_model to intialize the weights
        """
        if self.params['network']['init_weights'] == 'kaiming':
            nn.init.kaiming_normal_(self.prediction.weight)
        self.src_embedding.init_weights()
        self.trg_embedding.init_weights()
        self.prediction.bias.data.fill_(0)

    def merge(self, src_emb, trg_emb):
        """
        Merge source and target embeddings
        *_emb : N, T_t, T_s, d
        """
        N, Tt, Ts, _ = src_emb.size()
        if self.merge_mode == 'concat':
            return torch.cat((src_emb, trg_emb), dim=3)
        elif self.merge_mode == 'product':
            return src_emb * trg_emb
        elif self.merge_mode == 'bilinear':
            X = []
            for t in range(Tt):
                e = trg_emb[:, t:t + 1, (0), :]
                w = self.bw.expand(N, -1).unsqueeze(-1)
                x = torch.bmm(w, e).transpose(1, 2)
                x = torch.bmm(src_emb[:, (0)], x).unsqueeze(1)
                X.append(x)
            return torch.cat(X, dim=1)
        elif self.merge_mode == 'multi-sim':
            X = []
            for k in range(self.sim_dim):
                w = self.bw[k].expand(N, -1, -1)
                x = torch.bmm(torch.bmm(trg_emb[:, :, (0)], w), src_emb[:,
                    (0)].transpose(1, 2)).unsqueeze(-1)
                X.append(x)
            return torch.cat(X, dim=-1)
        elif self.merge_mode == 'multi-sim2':
            X = []
            for n in range(N):
                x = torch.bmm(torch.bmm(trg_emb[n:n + 1, :, (0)].expand(
                    self.sim_dim, -1, -1), self.bw), src_emb[n:n + 1, (0)].
                    expand(self.sim_dim, -1, -1).transpose(1, 2)).unsqueeze(0)
                X.append(x)
            return torch.cat(X, dim=0).permute(0, 2, 3, 1)
        else:
            raise ValueError('Unknown merging mode')

    def forward(self, data_src, data_trg):
        src_emb = self.src_embedding(data_src)
        trg_emb = self.trg_embedding(data_trg)
        Ts = src_emb.size(1)
        Tt = trg_emb.size(1)
        src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        X = self.merge(src_emb, trg_emb)
        X = self._forward(X, data_src['lengths'])
        logits = F.log_softmax(self.prediction(self.prediction_dropout(X)),
            dim=2)
        return logits

    def _forward(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net(X)
        if track:
            X, attn = self.aggregator(X, src_lengths, track=True)
            return X, attn
        X = self.aggregator(X, src_lengths, track=track)
        return X

    def update(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net.update(X)
        attn = None
        if track:
            X, attn = self.aggregator(X, src_lengths, track=track)
        else:
            X = self.aggregator(X, src_lengths, track=track)
        return X, attn

    def track_update(self, data_src, kwargs={}):
        """
        Sample and return tracked activations
        Using update where past activations are discarded
        """
        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)
        max_length = int(kwargs.get('max_length_a', 0) * Ts + kwargs.get(
            'max_length_b', 50))
        trg_labels = torch.LongTensor([[self.bos_token] for i in range(
            batch_size)])
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        src_emb = src_emb.unsqueeze(1)
        src_emb_ = src_emb
        seq = []
        alphas = []
        aligns = []
        activ_aligns = []
        activs = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in range(max_length):
            X = self.merge(src_emb, trg_emb)
            Y, attn = self.update(X, data_src['lengths'], track=True)
            if attn[0] is not None:
                alphas.append(attn[0])
            aligns.append(attn[1])
            activ_aligns.append(attn[2])
            activs.append(attn[3][0])
            proj = self.prediction_dropout(Y[:, (-1), :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, (self.padding_idx)] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1)
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds, t
                ).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                unfinished = torch.add(torch.mul((next_preds == self.
                    eos_token).type_as(logits), -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        return seq, alphas, aligns, activ_aligns, activs

    def track(self, data_src, kwargs={}):
        """
        Sample and return tracked activations
        """
        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)
        max_length = int(kwargs.get('max_length_a', 0) * Ts + kwargs.get(
            'max_length_b', 50))
        trg_labels = torch.LongTensor([[self.bos_token] for i in range(
            batch_size)])
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        src_emb = src_emb.unsqueeze(1)
        src_emb_ = src_emb
        seq = []
        alphas = []
        aligns = []
        activ_aligns = []
        activs = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in range(max_length):
            X = self.merge(src_emb, trg_emb)
            Y, attn = self._forward(X, data_src['lengths'], track=True)
            if attn[0] is not None:
                alphas.append(attn[0])
            aligns.append(attn[1])
            activ_aligns.append(attn[2])
            activs.append(attn[3][0])
            proj = self.prediction_dropout(Y[:, (-1), :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, (self.padding_idx)] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1)
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds, t
                ).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                unfinished = torch.add(torch.mul((next_preds == self.
                    eos_token).type_as(logits), -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.trg_embedding.reset_buffers()
        return seq, alphas, aligns, activ_aligns, activs

    def sample_update(self, data_src, scorer, kwargs={}):
        """
        Sample in evaluation mode
        Using update where past activations are discarded
        """
        beam_size = kwargs.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(data_src, kwargs)
        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)
        max_length = int(kwargs.get('max_length_a', 0) * Ts + kwargs.get(
            'max_length_b', 50))
        trg_labels = torch.LongTensor([[self.bos_token] for i in range(
            batch_size)])
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        src_emb = src_emb.unsqueeze(1)
        src_emb_ = src_emb
        seq = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in range(max_length):
            X = self.merge(src_emb, trg_emb)
            Y, _ = self.update(X, data_src['lengths'])
            proj = self.prediction_dropout(Y[:, (-1), :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, (self.padding_idx)] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1)
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds, t
                ).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                unfinished = torch.add(torch.mul((next_preds == self.
                    eos_token).type_as(logits), -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        return seq, None

    def sample(self, data_src, scorer, kwargs={}):
        """
        Sample in evaluation mode
        """
        beam_size = kwargs.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(data_src, kwargs)
        batch_size = data_src['labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)
        max_length = int(kwargs.get('max_length_a', 0) * Ts + kwargs.get(
            'max_length_b', 50))
        trg_labels = torch.LongTensor([[self.bos_token] for i in range(
            batch_size)])
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        src_emb = src_emb.unsqueeze(1)
        src_emb_ = src_emb
        seq = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in range(max_length):
            X = self.merge(src_emb, trg_emb)
            Y = self._forward(X, data_src['lengths'])
            proj = self.prediction_dropout(Y[:, (-1), :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, (self.padding_idx)] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1)
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds, t
                ).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                unfinished = torch.add(torch.mul((next_preds == self.
                    eos_token).type_as(logits), -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        return seq, None

    def sample_beam(self, data_src, kwargs={}):
        beam_size = kwargs['beam_size']
        src_labels = data_src['labels']
        src_lengths = data_src['lengths']
        batch_size = src_labels.size(0)
        beam = [Beam(beam_size, kwargs) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        Ts = src_labels.size(1)
        max_length = int(kwargs.get('max_length_a', 0) * Ts + kwargs.get(
            'max_length_b', 50))
        src_labels = src_labels.repeat(beam_size, 1)
        src_lengths = src_lengths.repeat(beam_size, 1)
        for t in range(max_length):
            src_emb = self.src_embedding({'labels': src_labels, 'lengths':
                None}).unsqueeze(1).repeat(1, t + 1, 1, 1)
            trg_labels_t = torch.stack([b.get_current_state() for b in beam if
                not b.done]).t().contiguous().view(-1, 1)
            if t:
                trg_labels = torch.cat((trg_labels, trg_labels_t), dim=1)
            else:
                trg_labels = trg_labels_t
            trg_emb = self.trg_embedding({'labels': trg_labels, 'lengths':
                None}).unsqueeze(2).repeat(1, 1, Ts, 1)
            X = self.merge(src_emb, trg_emb)
            Y = self._forward(X, src_lengths)
            proj = self.prediction_dropout(Y[:, (-1), :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            word_lk = logits.view(beam_size, remaining_sents, -1).transpose(
                0, 1).contiguous()
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue
                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], t):
                    active += [b]
                trg_labels_prev = trg_labels.view(beam_size,
                    remaining_sents, t + 1)
                trg_labels = trg_labels_prev[beam[b].get_current_origin()
                    ].view(-1, t + 1)
            if not active:
                break
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                view = t.data.contiguous().view(beam_size, remaining_sents,
                    *t.size()[1:])
                new_size = list(view.size())
                new_size[1] = new_size[1] * len(active_idx) // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result.view(-1, result.size(-1))
            src_labels = update_active(src_labels)
            src_lengths = update_active(src_lengths)
            trg_labels = update_active(trg_labels)
            remaining_sents = len(active)
        allHyp, allScores = [], []
        n_best = 1
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = beam[b].get_hyp(ks[0])
            allHyp += [hyps]
        return allHyp, allScores


class Pervasive_Parallel(nn.DataParallel):
    """
    Wrapper for parallel training
    """

    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
        special_tokens):
        model = Pervasive(jobname, params, src_vocab_size, trg_vocab_size,
            special_tokens)
        nn.DataParallel.__init__(self, model)
        self.logger = logging.getLogger(jobname)
        self.version = 'conv'
        self.params = params
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.pad_token = special_tokens['PAD']
        self.bos_token = special_tokens['BOS']
        self.eos_token = special_tokens['EOS']
        self.kernel_size = max(list(itertools.chain.from_iterable(params[
            'network']['kernels'])))

    def init_weights(self):
        self.module.init_weights()

    def _forward(self, X, src_lengths=None):
        return self.module._forward(self, X, src_lengths)

    def update(self, X, src_lengths=None):
        return self.module.update(X, src_lengths)

    def sample(self, data_src, scorer=None, kwargs={}):
        return self.module.sample(data_src, scorer, kwargs)


class MaxAttention(nn.Module):

    def __init__(self, params, in_channels):
        super(MaxAttention, self).__init__()
        self.in_channels = in_channels
        self.attend = nn.Linear(in_channels, 1)
        self.dropout = params['attention_dropout']
        self.scale_ctx = params.get('scale_ctx', 1)
        if params['nonlin'] == 'tanh':
            self.nonlin = F.tanh
        elif params['nonlin'] == 'relu':
            self.nonlin = F.relu
        else:
            self.nonlin = lambda x: x
        if params['first_aggregator'] == 'max':
            self.max = max_code
        elif params['first_aggregator'] == 'truncated-max':
            self.max = truncated_max
        elif params['first_aggregator'] == 'skip':
            self.max = None
        else:
            raise ValueError('Unknown mode for first aggregator ', params[
                'first_aggregator'])

    def forward(self, X, src_lengths, track=False, *args):
        if track:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            context = alphas.expand_as(Xatt) * Xatt
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            if self.max is not None:
                Xpool, tracking = self.max(X, src_lengths, track=True)
                feat = torch.cat((Xpool, context), dim=1)
                return feat, (alphas[(0), (-1), :, (0)].data.cpu().numpy(),
                    *tracking[1:])
            else:
                return context
        else:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            context = alphas.expand_as(Xatt) * Xatt
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            if self.max is not None:
                Xpool = self.max(X, src_lengths)
                return torch.cat((Xpool, context), dim=1)
            else:
                return context


class Seq2Seq(nn.Module):

    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
        trg_specials):
        """Initialize model."""
        nn.Module.__init__(self)
        self.logger = logging.getLogger(jobname)
        self.version = 'seq2seq'
        self.params = params
        self.encoder = Encoder(params['encoder'], src_vocab_size)
        self.decoder = CondDecoder(params['decoder'], params['encoder'],
            trg_vocab_size, trg_specials)
        self.mapper_dropout = nn.Dropout(params['mapper']['dropout'])
        self.mapper = nn.Linear(self.encoder.size, self.decoder.size)

    def init_weights(self):
        """Initialize weights."""
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.mapper.bias.data.fill_(0)

    def map(self, source):
        """ map the source code to the decoder cell size """
        source['state'][0] = nn.Tanh()(self.mapper_dropout(self.mapper(
            source['state'][0])))
        return source

    def forward(self, data_src, data_trg):
        source = self.encoder(data_src)
        source = self.map(source)
        logits = self.decoder(source, data_trg)
        return logits

    def sample(self, source, kwargs={}):
        """
        Sample given source with keys:
            state - ctx - emb
        """
        return self.decoder.sample(source, kwargs)


class Transition(nn.Sequential):
    """
    Transiton btw dense blocks:
    BN > ReLU > Conv(k=1) to reduce the number of channels
    """

    def __init__(self, num_input_features, num_output_features, init_weights=0
        ):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        conv = nn.Conv2d(num_input_features, num_output_features,
            kernel_size=1, bias=False)
        if init_weights == 'manual':
            std = sqrt(2 / num_input_features)
            conv.weight.data.normal_(0, std)
        self.add_module('conv', conv)

    def forward(self, x, *args):
        return super(Transition, self).forward(x)


class Transition2(nn.Sequential):
    """
    Transiton btw dense blocks:
    ReLU > Conv(k=1) to reduce the number of channels

    """

    def __init__(self, num_input_features, num_output_features):
        super(Transition2, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, bias=False))

    def forward(self, x, *args):
        return super(Transition2, self).forward(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_elbayadm_attn2d(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Aggregator(*[], **{'input_channls': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(MaskedConv1d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(AsymmetricMaskedConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MaskedConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(GatedConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(positional_encoding(*[], **{'num_units': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ChannelsNormalization(*[], **{'n_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(LayerNormalization(*[], **{'d_hid': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(LayerNorm(*[], **{'input_size': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(Transition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(Transition2(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

