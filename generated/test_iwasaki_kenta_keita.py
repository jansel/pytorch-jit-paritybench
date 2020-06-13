import sys
_module = sys.modules[__name__]
del sys
cnn_main = _module
text = _module
utils = _module
vision = _module
points_in_bbox = _module
sentence_sorting = _module
activation = _module
attention = _module
convolution = _module
nonlinear = _module
losses = _module
mahalanobis = _module
rnn_main = _module
classifiers = _module
encoders = _module
encoders = _module
encoders = _module
meta = _module

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


import torch.nn.functional as F


from torch import nn


import math


class GatedActivation(nn.Module):

    def __init__(self, num_channels):
        super(GatedActivation, self).__init__()
        self.kernel_size = 1
        self.weights = nn.Parameter(torch.FloatTensor(num_channels,
            num_channels, self.kernel_size * 2))

    def forward(self, x):
        """
        Conditional Image Generation with PixelCNN Decoders
        http://arxiv.org/abs/1606.05328

        1D gated activation unit that models the forget gates and
        real gates of an activation unit using convolutions.

        :param x: (batch size, # channels, height)
        :return: tanh(conv(Wr, x)) * sigmoid(conv(Wf, x))
        """
        real_gate_weights, forget_gate_weights = self.weights.split(self.
            kernel_size, dim=2)
        real_gate_weights = real_gate_weights.contiguous()
        forget_gate_weights = forget_gate_weights.contiguous()
        real_gate = F.tanh(F.conv1d(input=x, weight=real_gate_weights,
            stride=1))
        forget_gate = F.sigmoid(F.conv1d(input=x, weight=
            forget_gate_weights, stride=1))
        return real_gate * forget_gate


class BahdanauAttention(nn.Module):
    """
    Applies inner dot-product to the last hidden state of a RNN.
    * contains additional modes for being applied in various ways
    to bidirectional RNN models.

    "Neural Machine Translation by Jointly Learning to Align and Translate"
    https://arxiv.org/abs/1409.0473

    Mode 0: Single projection projection for left & right hidden states.
    Mode 1: Independent projection projections for left & right hidden states.
    Mode 2: Concatenate the hidden neurons of both the left & right hidden states and
        pass them through a single projection.
    """

    def __init__(self, hidden_size, mode=0):
        super(BahdanauAttention, self).__init__()
        self.mode = mode
        if mode == 0:
            self.projection = nn.Linear(hidden_size, hidden_size)
        elif mode == 1:
            self.left_projection = nn.Linear(hidden_size, hidden_size)
            self.right_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, *hidden_states):
        if len(hidden_states) == 1:
            hidden_state = hidden_states[0]
            return F.softmax(F.tanh(self.projection(hidden_state))
                ) * hidden_state
        elif len(hidden_states) == 2:
            left_hidden_state, right_hidden_state = hidden_states
            if self.mode == 0 or self.mode == 1:
                if self.mode == 0:
                    left_attention_weights = F.softmax(F.tanh(self.
                        projection(left_hidden_state)))
                    right_attention_weights = F.softmax(F.tanh(self.
                        projection(right_hidden_state)))
                elif self.mode == 1:
                    left_attention_weights = F.softmax(F.tanh(self.
                        left_projection(left_hidden_state)))
                    right_attention_weights = F.softmax(F.tanh(self.
                        right_projection(right_hidden_state)))
                return (left_attention_weights * left_hidden_state, 
                    right_attention_weights * right_hidden_state)
            elif self.mode == 2:
                hidden_state = torch.cat([left_hidden_state,
                    right_hidden_state], dim=1)
                attention_weights = F.softmax(F.tanh(self.projection(
                    hidden_state)))
                return (attention_weights * left_hidden_state, 
                    attention_weights * right_hidden_state)


class LuongAttention(nn.Module):
    """
    Applies various alignments to the last decoder state of a seq2seq
    model based on all encoder hidden states.

    "Effective Approaches to Attention-based Neural Machine Translation"
    https://arxiv.org/abs/1508.04025

    As outlined by the paper, the three alignment methods available are:
    "dot", "general", and "concat". Inputs are expected to be time-major
    first. (seq. length, batch size, hidden dim.)

    Both encoder and decoders are expected to have the same hidden dim.
    as well, which is not specifically covered by the paper.

    Masking support is available. The masking variable should be
    (seq. length, batch size) with 1's representing words and 0's
    representing padded values.
    """

    def __init__(self, hidden_size, mode='general'):
        super(LuongAttention, self).__init__()
        self.mode = mode
        if mode == 'general':
            self.projection = nn.Parameter(torch.FloatTensor(hidden_size,
                hidden_size))
        elif mode == 'concat':
            self.reduction = nn.Parameter(torch.FloatTensor(hidden_size * 2,
                hidden_size))
            self.projection = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def forward(self, last_state, states, mask=None):
        sequence_length, batch_size, hidden_dim = states.size()
        last_state = last_state.unsqueeze(0).expand(sequence_length,
            batch_size, last_state.size(1))
        if self.mode == 'dot':
            energies = last_state * states
            energies = energies.sum(dim=2).squeeze()
        elif self.mode == 'general':
            expanded_projection = self.projection.expand(sequence_length, *
                self.projection.size())
            energies = last_state * states.bmm(expanded_projection)
            energies = energies.sum(dim=2).squeeze()
        elif self.mode == 'concat':
            expanded_reduction = self.reduction.expand(sequence_length, *
                self.reduction.size())
            expanded_projection = self.projection.expand(sequence_length, *
                self.projection.size())
            energies = F.tanh(torch.cat([last_state, states], dim=2).bmm(
                expanded_reduction))
            energies = energies.bmm(expanded_projection).squeeze()
        if type(mask) == torch.autograd.Variable:
            energies = energies + (mask == 0).float() * -10000
        attention_weights = F.softmax(energies)
        return attention_weights


class BilinearAttention(nn.Module):
    """
    Creates a bilinear transformation between a decoder hidden state
    and a sequence of encoder/decoder hidden states. Specifically used
    as a form of inter-attention for abstractive text summarization.

    "A Deep Reinforced Model for Abstractive Summarization"
    https://arxiv.org/abs/1705.04304
    https://einstein.ai/research/your-tldr-by-an-ai-a-deep-reinforced-model-for-abstractive-summarization

    Hidden state sequences alongside a given target hidden state
    are expected to be time-major first. (seq. length, batch size, hidden dim.)

    Encoder and decoder hidden states may have different hidden dimensions.
    """

    def __init__(self, hidden_size, encoder_dim=None):
        super(BilinearAttention, self).__init__()
        self.encoder_dim = hidden_size if encoder_dim is None else encoder_dim
        self.projection = nn.Parameter(torch.FloatTensor(hidden_size, self.
            encoder_dim))

    def forward(self, last_state, states):
        if len(states.size()) == 2:
            states = states.unsqueeze(0)
        sequence_length, batch_size, state_dim = states.size()
        transformed_last_state = last_state @ self.projection
        transformed_last_state = transformed_last_state.expand(sequence_length,
            batch_size, self.encoder_dim)
        transformed_last_state = transformed_last_state.transpose(0, 1
            ).contiguous()
        transformed_last_state = transformed_last_state.view(batch_size, -1)
        states = states.transpose(0, 1).contiguous()
        states = states.view(batch_size, -1)
        energies = transformed_last_state * states
        energies = energies.sum(dim=1)
        if self.encoder_dim is not None:
            attention_weights = torch.cat([torch.exp(energies[0]), F.
                softmax(energies[1:])], dim=0)
        else:
            attention_weights = F.softmax(energies)
        return attention_weights


class SeparableConv2d(nn.Module):
    """
    A depth-wise convolution followed by a point-wise convolution.
    WARNING: Very slow! Unoptimized for PyTorch.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels, kernel_size=3, stride=stride, padding=1, groups=
            in_channels, bias=False)
        self.batch_norm_in = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_out = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batch_norm_in(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.batch_norm_out(x)
        x = self.activation(x)
        return x


class CausalConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inputs):
        """
        A 1D dilated convolution w/ padding such that the output
        is the same size as the input.

        :param inputs: (batch size, # channels, height)
        :return: (batch size, # channels, height)
        """
        x = F.pad(inputs.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)


class EncoderCRF(nn.Module):
    """
    A conditional random field with its features provided by a bidirectional RNN
    (GRU by default). As of right now, the model only accepts a batch size of 1
    to represent model parameter updates as a result of stochastic gradient descent.

    Primarily used for part-of-speech tagging in NLP w/ state-of-the-art results.

    In essence a heavily cleaned up version of the article:
    http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

    "Bidirectional LSTM-CRF Models for Sequence Tagging"
    https://arxiv.org/abs/1508.01991

    :param sentence: (seq. length, 1, word embedding size)
    :param sequence (training only): Ground truth sequence label (seq. length)
    :return: Viterbi path decoding score, and sequence.
    """

    def __init__(self, start_tag_index, stop_tag_index, tag_size,
        embedding_dim, hidden_dim):
        super(EncoderCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.start_tag_index = start_tag_index
        self.stop_tag_index = stop_tag_index
        self.tag_size = tag_size
        self.encoder = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=1,
            bidirectional=True)
        self.tag_projection = nn.Linear(hidden_dim, self.tag_size)
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.
            tag_size))
        self.hidden = self.init_hidden()

    def to_scalar(self, variable):
        return variable.view(-1).data.tolist()[0]

    def argmax(self, vector, dim=1):
        _, index = torch.max(vector, dim)
        return self.to_scalar(index)

    def state_log_likelihood(self, scores):
        max_score = scores.max()
        max_scores = max_score.unsqueeze(0).expand(*scores.size())
        return max_score + torch.log(torch.sum(torch.exp(scores - max_scores)))

    def init_hidden(self):
        return torch.autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, features):
        energies = torch.Tensor(1, self.tag_size).fill_(-10000.0)
        energies[0][self.start_tag_index] = 0.0
        energies = torch.autograd.Variable(energies)
        for feature in features:
            best_path = []
            next_state_scores = energies.expand(*self.transitions.size()
                ) + self.transitions + feature.unsqueeze(0).expand(*self.
                transitions.size())
            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index].unsqueeze(0)
                best_path.append(self.state_log_likelihood(
                    next_possible_states))
            energies = torch.cat(best_path).view(1, -1)
        terminal_energy = energies + self.transitions[self.stop_tag_index]
        return self.state_log_likelihood(terminal_energy)

    def encode(self, sentence):
        self.hidden = self.init_hidden()
        outputs, self.hidden = self.encoder(sentence, self.hidden)
        tag_energies = self.tag_projection(outputs.squeeze())
        return tag_energies

    def _score_sentence(self, features, tags):
        score = torch.autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.start_tag_index]), tags])
        for index, feature in enumerate(features):
            score = score + self.transitions[tags[index + 1], tags[index]
                ] + feature[tags[index + 1]]
        score = score + self.transitions[self.stop_tag_index, tags[-1]]
        return score

    def viterbi_decode(self, features):
        backpointers = []
        energies = torch.Tensor(1, self.tag_size).fill_(-10000.0)
        energies[0][self.start_tag_index] = 0
        energies = torch.autograd.Variable(energies)
        for feature in features:
            backtrack = []
            best_path = []
            next_state_scores = energies.expand(*self.transitions.size()
                ) + self.transitions
            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index]
                best_candidate_state = self.argmax(next_possible_states, dim=0)
                backtrack.append(best_candidate_state)
                best_path.append(next_possible_states[best_candidate_state])
            energies = (torch.cat(best_path) + feature).view(1, -1)
            backpointers.append(backtrack)
        terminal_energy = energies + self.transitions[self.stop_tag_index]
        best_candidate_state = self.argmax(terminal_energy)
        path_score = terminal_energy[0][best_candidate_state]
        best_path = [best_candidate_state]
        for backtrack in reversed(backpointers):
            best_candidate_state = backtrack[best_candidate_state]
            best_path.append(best_candidate_state)
        best_path.reverse()
        best_path = best_path[1:]
        return path_score, best_path

    def loss(self, sentence, tags):
        features = self.encode(sentence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        features = self.encode(sentence)
        viterbi_score, best_tag_sequence = self.viterbi_decode(features)
        return viterbi_score, best_tag_sequence


class MixtureDensityNetwork(nn.Module):

    def __init__(self, input_dim=1, hidden_size=50, num_mixtures=20):
        super(MixtureDensityNetwork, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_size)
        self.mean_projection = nn.Linear(hidden_size, num_mixtures)
        self.std_projection = nn.Linear(hidden_size, num_mixtures)
        self.weights_projection = nn.Linear(hidden_size, num_mixtures)

    def forward(self, x):
        """
        A model for non-linear data that works off of mixing multiple Gaussian
        distributions together. Uses linear projections of a given input to generate
        a set of N Gaussian models' mixture components, means and standard deviations.

        :param x: (num. samples, input dim.)
        :return: Mixture components, means, and standard deviations
            in the form (num. samples, num. mixtures)
        """
        x = F.tanh(self.projection(x))
        weights = F.softmax(self.weights_projection(x))
        means = self.mean_projection(x)
        stds = torch.exp(self.std_projection(x))
        return weights, means, stds


class MixtureDensityLoss(nn.Module):

    def __init__(self):
        super(MixtureDensityLoss, self).__init__()

    def forward(self, y, weights, mean, std):
        """
        Presents a maximum a-priori objective for a set of predicted means, mixture components,
        and standard deviations to model a given ground-truth 'y'. Modeled using negative log
        likelihood.

        :param y: Non-linear target.
        :param weights: Predicted mixture components.
        :param mean: Predicted mixture means.
        :param std: Predicted mixture standard deviations.
        :return:
        """
        normalization = 1.0 / (2.0 * math.pi) ** 0.5
        gaussian_sample = (y.expand_as(mean) - mean) * torch.reciprocal(std)
        gaussian_sample = normalization * torch.reciprocal(std) * torch.exp(
            -0.5 * gaussian_sample ** 2)
        return -torch.mean(torch.log(torch.sum(weights * gaussian_sample,
            dim=1)))


class MahalanobisMetricLoss(nn.Module):

    def __init__(self, margin=0.6, extra_margin=0.04):
        super(MahalanobisMetricLoss, self).__init__()
        self.margin = margin
        self.extra_margin = extra_margin

    def forward(self, outputs, targets):
        """
        :param outputs: Outputs from a network. (sentence_batch size, # features)
        :param targets: Target labels. (sentence_batch size, 1)
        :param margin: Minimum distance margin between contrasting sample pairs.
        :param extra_margin: Extra acceptable margin.
        :return: Loss and accuracy. Loss is a variable which may have a backward pass performed.
        """
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss
        loss = torch.autograd.Variable(loss)
        batch_size = outputs.size(0)
        magnitude = (outputs ** 2).sum(1).expand(batch_size, batch_size)
        squared_matrix = outputs.mm(torch.t(outputs))
        mahalanobis_distances = F.relu(magnitude + torch.t(magnitude) - 2 *
            squared_matrix).sqrt()
        neg_mask = targets.expand(batch_size, batch_size)
        neg_mask = neg_mask - neg_mask.transpose(0, 1) != 0
        num_pairs = (1 - neg_mask).sum()
        num_pairs = (num_pairs - batch_size) / 2
        num_pairs = num_pairs.data[0]
        negative_threshold = mahalanobis_distances[neg_mask].sort()[0][
            num_pairs].data[0]
        num_right, num_wrong = 0, 0
        for row in range(batch_size):
            for column in range(batch_size):
                x_label = targets[row].data[0]
                y_label = targets[column].data[0]
                mahalanobis_distance = mahalanobis_distances[row, column]
                euclidian_distance = torch.dist(outputs[row], outputs[column])
                if x_label == y_label:
                    if mahalanobis_distance.data[0
                        ] > self.margin - self.extra_margin:
                        loss += mahalanobis_distance - (self.margin - self.
                            extra_margin)
                    if euclidian_distance.data[0] < self.margin:
                        num_right += 1
                    else:
                        num_wrong += 1
                else:
                    if (mahalanobis_distance.data[0] < self.margin + self.
                        extra_margin and mahalanobis_distance.data[0] <
                        negative_threshold):
                        loss += (self.margin + self.extra_margin -
                            mahalanobis_distance)
                    if euclidian_distance.data[0] < self.margin:
                        num_wrong += 1
                    else:
                        num_right += 1
        accuracy = num_right / (num_wrong + num_right)
        return loss / (2 * num_pairs), accuracy


class HierarchialNetwork1D(nn.Module):
    """
    A shallow 1D CNN text classification model.
    Sequences are assumed to be 3D tensors (sequence length, sentence_batch size, word dim.)
    """

    def __init__(self, embed_dim, hidden_dim=64):
        super(HierarchialNetwork1D, self).__init__()
        self.layers = nn.ModuleList()
        first_block = nn.Sequential(nn.Conv1d(in_channels=embed_dim,
            out_channels=hidden_dim, kernel_size=3, padding=1), nn.ReLU(
            inplace=True), nn.BatchNorm1d(hidden_dim))
        self.layers.append(first_block)
        for layer_index in range(4):
            conv_block = nn.Sequential(nn.Conv1d(in_channels=hidden_dim,
                out_channels=hidden_dim, kernel_size=3, padding=1), nn.ReLU
                (inplace=True), nn.BatchNorm1d(hidden_dim))
            self.layers.append(conv_block)

    @staticmethod
    def get_output_size(hidden_dim):
        return hidden_dim * 5

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(F.max_pool1d(x, kernel_size=x.size(2)).
                squeeze())
        features = torch.cat(feature_maps, dim=1)
        return features


class BidirectionalEncoder(nn.Module):

    def __init__(self, embed_dim=50, hidden_dim=300, num_layers=4, dropout=
        0.1, rnn=nn.GRU, pooling_mode='max'):
        super(BidirectionalEncoder, self).__init__()
        self.pooling_mode = pooling_mode
        self.encoder = rnn(input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, bidirectional=True, dropout=dropout)

    @staticmethod
    def get_output_size(hidden_dim):
        return hidden_dim * 2

    def forward(self, x):
        """
        A bidirectional RNN encoder. Has support for global max/average pooling.

        :param x: A tuple of Variable's representing padded sentence tensor batch
            [seq. length, batch size, embed. size] and sentence lengths.
        :return: Global max/average pooled embedding from bidirectional RNN encoder of [batch_size, hidden_size]
        """
        sentences, sentence_lengths = x
        sorted_sentence_lengths, sort_indices = torch.sort(sentence_lengths,
            dim=0, descending=True)
        _, unsort_indices = torch.sort(sort_indices, dim=0)
        sorted_sentence_lengths = sorted_sentence_lengths.data
        sorted_sentences = sentences.index_select(1, sort_indices)
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences,
            sorted_sentence_lengths.clone().cpu().numpy())
        encoder_outputs = self.encoder(packed_sentences)[0]
        encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0]
        encoder_outputs = encoder_outputs.index_select(1, unsort_indices)
        encoder_outputs = encoder_outputs.transpose(0, 2).transpose(0, 1)
        if self.pooling_mode == 'max':
            encoder_outputs = F.max_pool1d(encoder_outputs, kernel_size=
                encoder_outputs.size(2))
        elif self.pooling_mode == 'avg':
            encoder_outputs = F.avg_pool1d(encoder_outputs, kernel_size=
                encoder_outputs.size(2))
        encoder_outputs = encoder_outputs.squeeze()
        return encoder_outputs


class OmniglotEncoder(nn.Module):

    def __init__(self, feature_size=64):
        super(OmniglotEncoder, self).__init__()
        self.layers = nn.ModuleList()
        first_block = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            feature_size, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(feature_size), nn.LeakyReLU(inplace=True), nn.
            MaxPool2d(kernel_size=2))
        self.layers.append(first_block)
        for layer_index in range(3):
            block = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
                feature_size, kernel_size=3, stride=1, padding=1), nn.
                BatchNorm2d(feature_size), nn.LeakyReLU(inplace=True), nn.
                MaxPool2d(kernel_size=2))
            self.layers.append(block)
        self.fc = nn.Linear(feature_size, feature_size)

    def forward(self, input):
        """
        Matching Networks for One Shot Learning
        https://arxiv.org/abs/1606.04080

        A network specifically for embedding images from the Omniglot dataset.
        Primarily used with the TCML network.

        :param input: (batch size, # channels, height, width)
        :return: 64-dim embedding for a given image_embedding.
        """
        for layer in self.layers:
            input = layer(input)
        output = input.view(input.size(0), -1)
        output = self.fc(output)
        return output


class TemporalDenseBlock(nn.Module):

    def __init__(self, in_channels, hidden_size=128, dilation=1):
        super(TemporalDenseBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels=in_channels, out_channels=
            hidden_size, kernel_size=2, dilation=dilation)
        self.conv2 = CausalConv1d(in_channels=hidden_size, out_channels=
            hidden_size, kernel_size=2, dilation=dilation)
        self.conv3 = CausalConv1d(in_channels=hidden_size, out_channels=
            hidden_size, kernel_size=2, dilation=dilation)
        self.gate1 = GatedActivation(hidden_size)
        self.gate2 = GatedActivation(hidden_size)
        self.gate3 = GatedActivation(hidden_size)

    def forward(self, x):
        """
        A 1D dilated causal convolution dense block for TCML.
        Contains residual connections for the 2nd and 3rd convolution.

        :param x: (batch size, # channels, seq. length)
        :return: (batch size, # channels + 128, seq. length)
        """
        features = self.gate1(self.conv1(x))
        features = self.gate2(self.conv2(features) + features)
        features = self.gate3(self.conv3(features) + features)
        outputs = torch.cat([features, x], dim=1)
        return outputs


class TCML(nn.Module):

    def __init__(self, feature_dim, num_classes=3):
        super(TCML, self).__init__()
        self.dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        self.dense_blocks = nn.ModuleList([TemporalDenseBlock(feature_dim +
            128 * index, hidden_size=128, dilation=dilation) for index,
            dilation in enumerate(self.dilations)])
        self.conv1 = nn.Conv1d(in_channels=feature_dim + 128 * len(self.
            dilations), out_channels=512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=num_classes,
            kernel_size=1, stride=1)

    def forward(self, inputs):
        """
        Meta-Learning with Temporal Convolutions
        https://arxiv.org/abs/1707.03141

        :param inputs: (batch size, # channels, height)
        :return: (batch size, num. classes, height)
        """
        for index, block in enumerate(self.dense_blocks):
            features = block(inputs if index == 0 else features)
        features = F.relu(self.conv1(features), inplace=True)
        features = self.conv2(features)
        return features


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_iwasaki_kenta_keita(_paritybench_base):
    pass
    def test_000(self):
        self._check(GatedActivation(*[], **{'num_channels': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BahdanauAttention(*[], **{'hidden_size': 4}), [], {})

    @_fails_compile()
    def test_002(self):
        self._check(LuongAttention(*[], **{'hidden_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(CausalConv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    def test_005(self):
        self._check(MixtureDensityNetwork(*[], **{}), [torch.rand([1, 1])], {})

    def test_006(self):
        self._check(MixtureDensityLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(HierarchialNetwork1D(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(TemporalDenseBlock(*[], **{'in_channels': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_009(self):
        self._check(TCML(*[], **{'feature_dim': 4}), [torch.rand([4, 4, 64])], {})

