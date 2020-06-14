import sys
_module = sys.modules[__name__]
del sys
data = _module
generic_model = _module
main = _module
model = _module
rescore = _module
utils = _module
vocab = _module
nce = _module
alias_multinomial = _module
index_gru = _module
index_linear = _module
nce_loss = _module
sample = _module
setup = _module
test_evaluation = _module

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


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataloader import DataLoader


from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn


import time


import math


import torch.optim as optim


from math import isclose


import torch.nn.functional as F


def get_mask(lengths, cut_tail=0, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    assert lengths.min() >= cut_tail
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(
        lengths.unsqueeze(1))
    return mask


class GenModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, criterion, dropout=0.2):
        super(GenModel, self).__init__()
        self.criterion = criterion

    def forward(self, input, target, length):
        mask = get_mask(length.data, cut_tail=0)
        effective_target = target[:, 1:].contiguous()
        loss = self.criterion(effective_target, input)
        loss = torch.masked_select(loss, mask)
        return loss.mean()


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout,
            batch_first=True)
        self.proj = nn.Linear(nhid, ninp)
        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input):
        """Serves as the encoder and recurrent layer"""
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb)
        output = self.proj(output)
        output = self.drop(output)
        return output

    def forward(self, input, target, length):
        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)
        return loss.mean()


class AliasMultinomial(torch.nn.Module):
    """Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()
        assert isclose(probs.sum().item(), 1
            ), 'The noise distribution must sum to 1'
        cpu_probs = probs.cpu()
        K = len(probs)
        self_prob = [0] * K
        self_alias = [0] * K
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K * prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self_alias[small] = large
            self_prob[large] = self_prob[large] - 1.0 + self_prob[small]
            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self_prob[last_one] = 1
        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)
        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)
        return (oq + oj).view(size)


BACKOFF_PROB = 1e-10


class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        noise_ratio: $rac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self, noise, noise_ratio=100, norm_term='auto', reduction=
        'elementwise_mean', per_word=False, loss_type='nce'):
        super(NCELoss, self).__init__()
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)
        self.noise_ratio = noise_ratio
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """
        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':
            noise_samples = self.get_noise(batch, max_len)
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.
                view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)
                ].view_as(target)
            logit_target_in_model, logit_noise_in_model = self._get_logit(
                target, noise_samples, *args, **kwargs)
            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(logit_target_in_model,
                        logit_noise_in_model, logit_noise_in_noise,
                        logit_target_in_noise)
                else:
                    loss = -logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(logit_target_in_model,
                    logit_noise_in_model, logit_noise_in_noise,
                    logit_target_in_noise)
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(logit_target_in_model,
                    logit_noise_in_model, logit_noise_in_noise,
                    logit_target_in_noise)
                loss += 0.5 * self.sampled_softmax_loss(logit_target_in_model,
                    logit_noise_in_model, logit_noise_in_noise,
                    logit_target_in_noise)
            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError('loss type {} not implemented at {}'
                    .format(self.loss_type, current_stage))
        else:
            loss = self.ce_loss(target, *args, **kwargs)
        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""
        noise_size = batch_size, max_len, self.noise_ratio
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*
                noise_size)
        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise

        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """
        target_logit, noise_logit = self.get_score(target_idx, noise_idx, *
            args, **kwargs)
        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, *args, **kwargs):
        """Get the target and noise score

        Usually logits are used as score.
        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        raise NotImplementedError()

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss

        The returned loss should be of the same size of `target`

        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class

        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, logit_target_in_model, logit_noise_in_model,
        logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """
        logit_model = torch.cat([logit_target_in_model.unsqueeze(2),
            logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2),
            logit_noise_in_noise], dim=2)
        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)
        label = torch.zeros_like(logit_model)
        label[:, :, (0)] = 1
        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model,
        logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        logits = torch.cat([logit_target_in_model.unsqueeze(2),
            logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2),
            logit_noise_in_noise], dim=2)
        logits = logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        loss = self.ce(logits.view(-1, logits.size(-1)), labels.view(-1)
            ).view_as(labels)
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Stonesjtu_Pytorch_NCE(_paritybench_base):
    pass
