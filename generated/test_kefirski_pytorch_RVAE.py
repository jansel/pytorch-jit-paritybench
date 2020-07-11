import sys
_module = sys.modules[__name__]
del sys
master = _module
model = _module
decoder = _module
encoder = _module
rvae = _module
sample = _module
selfModules = _module
embedding = _module
highway = _module
neg = _module
tdnn = _module
train = _module
train_word_embeddings = _module
utils = _module
batch_loader = _module
functional = _module
parameters = _module
visualize_word_embeddings = _module

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


import torch as t


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.autograd import Variable


from torch.nn import Parameter


from torch.optim import Adam


from torch.optim import SGD


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def fold(f, l, a):
    return a if len(l) == 0 else fold(f, l[1:], f(a, l[0]))


def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size, hidden_size=self.params.decoder_rnn_size, num_layers=self.params.decoder_num_layers, batch_first=True)
        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def forward(self, decoder_input, z, drop_prob, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        assert parameters_allocation_check(self), 'Invalid CUDA options. Parameters should be allocated in the same memory'
        [batch_size, seq_len, _] = decoder_input.size()
        """
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        """
        decoder_input = F.dropout(decoder_input, drop_prob)
        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)
        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)
        return result, final_state


class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.nonlinear):
            self._add_to_parameters(module.parameters(), 'nonlinear_module_{}'.format(i))
        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), 'linear_module_{}'.format(i))
        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), 'gate_module_{}'.format(i))
        self.f = f

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]

        :return: tensor with shape of [batch_size, size]

        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)


class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)
        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth, hidden_size=self.params.encoder_rnn_size, num_layers=self.params.encoder_num_layers, batch_first=True, bidirectional=True)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """
        [batch_size, seq_len, embed_size] = input.size()
        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)
        assert parameters_allocation_check(self), 'Invalid CUDA options. Parameters should be allocated in the same memory'
        """ Unfold rnn with zero initial state and get its final state from the last layer
        """
        _, (_, final_state) = self.rnn(input)
        final_state = final_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)
        return final_state


class TDNN(nn.Module):

    def __init__(self, params):
        super(TDNN, self).__init__()
        self.params = params
        self.kernels = [Parameter(t.Tensor(out_dim, self.params.char_embed_size, kW).uniform_(-1, 1)) for kW, out_dim in params.kernels]
        self._add_to_parameters(self.kernels, 'TDNN_kernel')

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]

        :return: tensor with shape [batch_size, max_seq_len, depth_sum]

        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """
        input_size = x.size()
        input_size_len = len(input_size)
        assert input_size_len == 4, 'Wrong input rang, must be equal to 4, but {} found'.format(input_size_len)
        [batch_size, seq_len, _, embed_size] = input_size
        assert embed_size == self.params.char_embed_size, 'Wrong embedding size, must be equal to {}, but {} found'.format(self.params.char_embed_size, embed_size)
        x = x.view(-1, self.params.max_word_len, self.params.char_embed_size).transpose(1, 2).contiguous()
        xs = [F.tanh(F.conv1d(x, kernel)) for kernel in self.kernels]
        xs = [x.max(2)[0].squeeze(2) for x in xs]
        x = t.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)
        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)


class Embedding(nn.Module):

    def __init__(self, params, path='../../../'):
        super(Embedding, self).__init__()
        self.params = params
        word_embed = np.load(path + 'data/word_embeddings.npy')
        self.word_embed = nn.Embedding(self.params.word_vocab_size, self.params.word_embed_size)
        self.char_embed = nn.Embedding(self.params.char_vocab_size, self.params.char_embed_size)
        self.word_embed.weight = Parameter(t.from_numpy(word_embed).float(), requires_grad=False)
        self.char_embed.weight = Parameter(t.Tensor(self.params.char_vocab_size, self.params.char_embed_size).uniform_(-1, 1))
        self.TDNN = TDNN(self.params)

    def forward(self, word_input, character_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :param character_input: [batch_size, seq_len, max_word_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size + sum_depth]
        """
        assert word_input.size()[:2] == character_input.size()[:2], 'Word input and character input must have the same sizes, but {} and {} found'.format(word_input.size(), character_input.size())
        [batch_size, seq_len] = word_input.size()
        word_input = self.word_embed(word_input)
        character_input = character_input.view(-1, self.params.max_word_len)
        character_input = self.char_embed(character_input)
        character_input = character_input.view(batch_size, seq_len, self.params.max_word_len, self.params.char_embed_size)
        character_input = self.TDNN(character_input)
        result = t.cat([word_input, character_input], 2)
        return result


def kld_coef(i):
    import math
    return (math.tanh((i - 3500) / 1000) + 1) / 2


class RVAE(nn.Module):

    def __init__(self, params):
        super(RVAE, self).__init__()
        self.params = params
        self.embedding = Embedding(self.params, '')
        self.encoder = Encoder(self.params)
        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.decoder = Decoder(self.params)

    def forward(self, drop_prob, encoder_word_input=None, encoder_character_input=None, decoder_word_input=None, decoder_character_input=None, z=None, initial_state=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        assert parameters_allocation_check(self), 'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda
        assert z is None and fold(lambda acc, parameter: acc and parameter is not None, [encoder_word_input, encoder_character_input, decoder_word_input], True) or z is not None and decoder_word_input is not None, 'Invalid input. If z is None then encoder and decoder inputs should be passed as arguments'
        if z is None:
            """ Get context from encoder and sample z ~ N(mu, std)
            """
            [batch_size, _] = encoder_word_input.size()
            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            context = self.encoder(encoder_input)
            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)
            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z
            z = z * std + mu
            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None
        decoder_input = self.embedding.word_embed(decoder_word_input)
        out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state)
        return out, final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):

        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [(var if use_cuda else var) for var in input]
            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
            logits, _, kld = self(dropout, encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, z=None)
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)
            loss = 79 * cross_entropy + kld_coef(i) * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return cross_entropy, kld, kld_coef(i)
        return train

    def validater(self, batch_loader):

        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [(var if use_cuda else var) for var in input]
            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
            logits, _, kld = self(0.0, encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, z=None)
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)
            return cross_entropy, kld
        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda):
        seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed
        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)
        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())
        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input, decoder_character_input
        result = ''
        initial_state = None
        for i in range(seq_len):
            logits, initial_state, _ = self(0.0, None, None, decoder_word_input, decoder_character_input, seed, initial_state)
            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)
            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
            if word == batch_loader.end_token:
                break
            result += ' ' + word
            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])
            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())
            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input, decoder_character_input
        return result


class NEG_loss(nn.Module):

    def __init__(self, num_classes, embed_size):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. Embedding size
        """
        super(NEG_loss, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))
        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

    def forward(self, input_labes, out_labels, num_sampled):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size] of Long type
        :param num_sampled: An int. The number of sampled from noise examples

        :return: Loss estimation with shape of [batch_size]
            loss defined in Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality
            papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        """
        use_cuda = self.out_embed.weight.is_cuda
        [batch_size] = input_labes.size()
        input = self.in_embed(input_labes)
        output = self.out_embed(out_labels)
        noise = Variable(t.Tensor(batch_size, num_sampled).uniform_(0, self.num_classes - 1).long())
        if use_cuda:
            noise = noise
        noise = self.out_embed(noise).neg()
        log_target = (input * output).sum(1).squeeze().sigmoid().log()
        """ ∑[batch_size, num_sampled, embed_size] * [batch_size, embed_size, 1] ->
            ∑[batch_size, num_sampled] -> [batch_size] """
        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Highway,
     lambda: ([], {'size': 4, 'num_layers': 1, 'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NEG_loss,
     lambda: ([], {'num_classes': 4, 'embed_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64), 4], {}),
     False),
]

class Test_kefirski_pytorch_RVAE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

