import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
dataset = _module
fold_dataset = _module
main4model = _module
main_boost = _module
main_fold = _module
search_all = _module
search_aug_noMultimodel_weight1 = _module
search_multimodel = _module
search_paris = _module
search_test = _module
search_weight5 = _module
searchstack = _module
searchstack_new = _module
test = _module
test_aug_multimodel = _module
test_stack = _module
main = _module
BasicModule = _module
CNNText_inception = _module
FastText3 = _module
LSTMText = _module
MultiCNNTextBNDeep = _module
MultiModelAll = _module
MultiModelAll2 = _module
MultiModelAll4zhihu = _module
RCNN = _module
models = _module
alias_multinomial = _module
loss = _module
nce = _module
rep = _module
embedding2matrix = _module
label2id = _module
question2array = _module
ensamble = _module
graph2vec = _module
mer_csv = _module
search = _module
utils = _module
calculate_score = _module
optimizer = _module
visualize = _module

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


import torch as t


import time


import torch


import numpy as np


from torch import nn


from collections import OrderedDict


import torch.nn as nn


from torch.autograd import Variable


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, change_opt=True):
        None
        data = t.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
                self.opt.embedding_path = None
                self.__init__(self.opt)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self

    def save(self, name=None, new=False):
        prefix = 'checkpoints/' + self.model_name + '_' + self.opt.type_ + '_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name
        if new:
            data = {'opt': self.opt.state_dict(), 'd': self.state_dict()}
        else:
            data = self.state_dict()
        t.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = [p for p in self.parameters() if id(p) not in
            ignored_params]
        if lr2 is None:
            lr2 = lr1 * 0.5
        optimizer = t.optim.Adam([dict(params=base_params, weight_decay=
            weight_decay, lr=lr1), {'params': self.encoder.parameters(),
            'lr': lr2}])
        return optimizer


class Inception(nn.Module):

    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert co % 4 == 0
        cos = [co / 4] * 4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin,
            cos[0], 1, stride=1))]))
        self.branch2 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin,
            cos[1], 1)), ('norm1', nn.BatchNorm1d(cos[1])), ('relu1', nn.
            ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[1], cos[1], 3,
            stride=1, padding=1))]))
        self.branch3 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin,
            cos[2], 3, padding=1)), ('norm1', nn.BatchNorm1d(cos[2])), (
            'relu1', nn.ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[2],
            cos[2], 5, stride=1, padding=2))]))
        self.branch4 = nn.Sequential(OrderedDict([('conv3', nn.Conv1d(cin,
            cos[3], 3, stride=1, padding=1))]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1)
            )
        return result


class AliasMethod(object):
    """
        From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def draw(self, N):
        """
            Draw N samples from multinomial
        """
        K = self.alias.size(0)
        kk = torch.LongTensor(np.random.randint(0, K, size=N))
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj


class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        nhidden: hidden size of LSTM(a.k.a the output size)
        ntokens: vocabulary size
        noise: the distribution of noise
        noise_ratio: $rac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        decoder: the decoder matrix
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - decoder: :math:`(E, V)` where `E = embedding size`
    """

    def __init__(self, ntokens, nhidden, noise, noise_ratio=10, norm_term=9,
        size_average=True, decoder_weight=None):
        super(NCELoss, self).__init__()
        self.noise = noise
        self.alias = AliasMethod(noise)
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.ntokens = ntokens
        self.size_average = size_average
        self.decoder = IndexLinear(nhidden, ntokens)
        if decoder_weight:
            self.decoder.weight = decoder_weight

    def forward(self, input, target=None):
        """compute the loss with output and the desired target
        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.
        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`
        Return:
            the scalar NCELoss Variable ready for backward
        """
        length = target.size(0)
        if self.training:
            assert input.size(0) == target.size(0)
            noise_samples = self.alias.draw(self.noise_ratio).unsqueeze(0
                ).repeat(length, 1)
            data_prob, noise_in_data_probs = self._get_prob(input, target.
                data, noise_samples)
            noise_probs = Variable(self.noise[noise_samples.view(-1)].
                view_as(noise_in_data_probs))
            rnn_loss = torch.log(data_prob / (data_prob + self.noise_ratio *
                Variable(self.noise[target.data])))
            noise_loss = torch.sum(torch.log(self.noise_ratio * noise_probs /
                (noise_in_data_probs + self.noise_ratio * noise_probs)), 1)
            loss = -1 * torch.sum(rnn_loss + noise_loss)
        else:
            out = self.decoder(input, indices=target.unsqueeze(1))
            nll = out.sub(self.norm_term)
            loss = -1 * nll.sum()
        if self.size_average:
            loss = loss / length
        return loss

    def _get_prob(self, embedding, target_idx, noise_idx):
        """Get the NCE estimated probability for target and noise
        Shape:
            - Embedding: :math:`(N, E)`
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """
        embedding = embedding
        indices = Variable(torch.cat([target_idx.unsqueeze(1), noise_idx],
            dim=1))
        probs = self.decoder(embedding, indices)
        probs = probs.sub(self.norm_term).exp()
        return probs[:, (0)], probs[:, 1:]


class IndexLinear(nn.Linear):
    """A linear layer that only decodes the results of provided indices
    Args:
        input: the list of embedding
        indices: the indices of interests.
    Shape:
        - Input :math:`(N, in\\_features)`
        - Indices :math:`(N, 1+N_r)` where `max(M) <= N`
    Return:
        - out :math:`(N, 1+N_r)`
    """

    def forward(self, input, indices=None):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """
        if indices is None:
            return super(IndexLinear, self).forward(input)
        input = input.unsqueeze(1)
        target_batch = self.weight.index_select(0, indices.view(-1)).view(
            indices.size(0), indices.size(1), -1).transpose(1, 2)
        bias = self.bias.index_select(0, indices.view(-1)).view(indices.
            size(0), 1, indices.size(1))
        out = torch.baddbmm(1, bias, 1, input, target_batch)
        return out.squeeze()

    def reset_parameters(self):
        init_range = 0.1
        self.bias.data.fill_(0)
        self.weight.data.uniform_(-init_range, init_range)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_chenyuntc_PyTorchText(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(IndexLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

