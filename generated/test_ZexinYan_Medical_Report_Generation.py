import sys
_module = sys.modules[__name__]
del sys
clean_caption = _module
clean_tag = _module
debugger = _module
metric_performance = _module
pycocoevalcap = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
eval = _module
meteor = _module
rouge = _module
tokenizer = _module
ptbtokenizer = _module
clean_data = _module
sample = _module
tester = _module
trainer = _module
utils = _module
build_tag = _module
build_vocab = _module
callbacks = _module
dataset = _module
logger = _module
loss = _module
models = _module
models_debugger = _module
tcn = _module

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


import time


import torch


import torch.optim as optim


import torchvision.transforms as transforms


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import DataLoader


from torch.autograd import Variable


import numpy as np


import warnings


from collections import deque


from collections import OrderedDict


from collections import Iterable


from torch.utils.data import Dataset


from torchvision import transforms


from torch.nn.modules import loss


import torch.nn as nn


import torchvision


import torchvision.models as models


from torchvision.models.vgg import model_urls as vgg_model_urls


from torch.nn.utils import weight_norm


class WARPLoss(loss.Module):

    def __init__(self, num_labels=204):
        super(WARPLoss, self).__init__()
        self.rank_weights = [1.0 / 1]
        for i in range(1, num_labels):
            self.rank_weights.append(self.rank_weights[i - 1] + (1.0 / i + 1))

    def forward(self, input, target) ->object:
        """

        :rtype:
        :param input: Deep features tensor Variable of size batch x n_attrs.
        :param target: Ground truth tensor Variable of size batch x n_attrs.
        :return:
        """
        batch_size = target.size()[0]
        n_labels = target.size()[1]
        max_num_trials = n_labels - 1
        loss = 0.0
        for i in range(batch_size):
            for j in range(n_labels):
                if target[i, j] == 1:
                    neg_labels_idx = np.array([idx for idx, v in enumerate(target[i, :]) if v == 0])
                    neg_idx = np.random.choice(neg_labels_idx, replace=False)
                    sample_score_margin = 1 - input[i, j] + input[i, neg_idx]
                    num_trials = 0
                    while sample_score_margin < 0 and num_trials < max_num_trials:
                        neg_idx = np.random.choice(neg_labels_idx, replace=False)
                        num_trials += 1
                        sample_score_margin = 1 - input[i, j] + input[i, neg_idx]
                    r_j = np.floor(max_num_trials / num_trials)
                    weight = self.rank_weights[r_j]
                    for k in range(n_labels):
                        if target[i, k] == 0:
                            score_margin = 1 - input[i, j] + input[i, k]
                            loss += weight * torch.clamp(score_margin, min=0.0)
        return loss


class MultiLabelSoftmaxRegressionLoss(loss.Module):

    def __init__(self):
        super(MultiLabelSoftmaxRegressionLoss, self).__init__()

    def forward(self, input, target) ->object:
        return -1 * torch.sum(input * target)


class VisualFeatureExtractor(nn.Module):

    def __init__(self, pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.out_features = resnet.fc.in_features

    def forward(self, images) ->object:
        """

        :rtype: object
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features


class MLC(nn.Module):

    def __init__(self, classes=156, sementic_features_dim=512, fc_in_features=2048, k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()

    def forward(self, visual_features) ->object:
        """

        :rtype: object
        """
        tags = self.softmax(self.classifier(visual_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):

    def __init__(self, embed_size=512, hidden_size=512, visual_size=2048):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)
        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)
        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)
        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=10, momentum=0.1)
        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.bn_a_att = nn.BatchNorm1d(num_features=10, momentum=0.1)
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=0.1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, visual_features, semantic_features, h_sent) ->object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(visual_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))
        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, visual_features)
        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)
        ctx = self.bn_fc(self.W_fc(torch.cat([v_att, a_att], dim=1)))
        return ctx, v_att


class SentenceLSTM(nn.Module):

    def __init__(self, embed_size=512, hidden_size=512, num_layers=1):
        super(SentenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.W_t_h = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_t_ctx = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_stop_s_1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_stop_s = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_stop = nn.Linear(in_features=embed_size, out_features=2, bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_topic = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.W_topic_2 = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_topic_2 = nn.BatchNorm1d(num_features=1, momentum=0.1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, ctx, prev_hidden_state, states=None) ->object:
        """
        v2
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.bn_topic(self.W_topic(self.tanh(self.bn_t_h(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))))
        p_stop = self.bn_stop(self.W_stop(self.tanh(self.bn_stop_s(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))))
        return topic, p_stop, hidden_state, states


class WordLSTM(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions) ->object:
        """

        :rtype: object
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def val(self, features, start_tokens):
        samples = torch.zeros((np.shape(features)[0], self.n_max, self.vocab_size))
        samples[:, 0, start_tokens[0]] = 1
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings
        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            samples[:, i, :] = outputs
            predicted = torch.max(outputs, 1)[1]
            predicted = predicted.unsqueeze(1)
        return samples

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        sampled_ids[:, 0] = start_tokens.view(-1)
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings
        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids


class DenseNet121(nn.Module):

    def __init__(self, classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(in_features=num_in_features, out_features=classes, bias=True))

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(DenseNet161, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class DenseNet169(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(DenseNet169, self).__init__()
        self.model = torchvision.models.densenet169(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class DenseNet201(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(DenseNet201, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes), nn.Sigmoid())

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet18(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet34(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet50(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet101(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet152(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class VGG19(nn.Module):

    def __init__(self, classes=14, pretrained=True):
        super(VGG19, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=pretrained)
        self.model.classifier = nn.Sequential(self.__init_linear(in_features=25088, out_features=4096), nn.ReLU(), nn.Dropout(0.5), self.__init_linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout(0.5), self.__init_linear(in_features=4096, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class VGG(nn.Module):

    def __init__(self, tags_num):
        super(VGG, self).__init__()
        vgg_model_urls['vgg19'] = vgg_model_urls['vgg19'].replace('https://', 'http://')
        self.vgg19 = models.vgg19(pretrained=True)
        vgg19_classifier = list(self.vgg19.classifier.children())[:-1]
        self.classifier = nn.Sequential(*vgg19_classifier)
        self.fc = nn.Linear(4096, tags_num)
        self.fc.apply(self.init_weights)
        self.bn = nn.BatchNorm1d(tags_num, momentum=0.1)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            self.fc.weight.data.normal_(0, 0.1)
            self.fc.bias.data.fill_(0)

    def forward(self, images) ->object:
        """

        :rtype: object
        """
        visual_feats = self.vgg19.features(images)
        tags_classifier = visual_feats.view(visual_feats.size(0), -1)
        tags_classifier = self.bn(self.fc(self.classifier(tags_classifier)))
        return tags_classifier


class InceptionV3(nn.Module):

    def __init__(self, classes=156, pretrained=True):
        super(InceptionV3, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(self.__init_linear(in_features=num_in_features, out_features=classes))

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class CheXNetDenseNet121(nn.Module):

    def __init__(self, classes=14, pretrained=True):
        super(CheXNetDenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(in_features=num_in_features, out_features=classes, bias=True), nn.Sigmoid())

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class CheXNet(nn.Module):

    def __init__(self, classes=156):
        super(CheXNet, self).__init__()
        self.densenet121 = CheXNetDenseNet121(classes=14)
        self.densenet121 = torch.nn.DataParallel(self.densenet121)
        self.densenet121.load_state_dict(torch.load('./models/CheXNet.pth.tar')['state_dict'])
        self.densenet121.module.densenet121.classifier = nn.Sequential(self.__init_linear(1024, classes), nn.Sigmoid())

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) ->object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, pretrained=True):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.1)
        self.__init_weights()

    def __init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, images) ->object:
        """

        :rtype: object
        """
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, n_max=50):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions) ->object:
        """

        :rtype: object
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings.unsqueeze(1)
        for i in range(self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x) ->object:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=False)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=False)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x) ->object:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x) ->object:
        return self.network(x)


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, inputs) ->object:
        y = self.tcn.forward(inputs)
        output = self.linear(y[:, :, -1])
        return output


class SentenceTCN(nn.Module):

    def __init__(self, input_channel=10, embed_size=512, output_size=512, nhid=512, levels=8, kernel_size=2, dropout=0):
        super(SentenceTCN, self).__init__()
        channel_sizes = [nhid] * levels
        self.tcn = TCN(input_size=input_channel, output_size=output_size, num_channels=channel_sizes, kernel_size=kernel_size, dropout=dropout)
        self.W_t_h = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_t_ctx = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop_s_1 = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop_s = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop = nn.Linear(in_features=embed_size, out_features=2, bias=True)
        self.t_w = nn.Linear(in_features=5120, out_features=2, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, ctx, prev_output) ->object:
        """

        :rtype: object
        """
        output = self.tcn.forward(ctx)
        topic = self.tanh(self.W_t_h(output) + self.W_t_ctx(ctx[:, -1, :]).squeeze(1))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_output) + self.W_stop_s(output)))
        return topic, p_stop, output


class WordTCN(nn.Module):

    def __init__(self, input_channel=11, vocab_size=1000, embed_size=512, output_size=512, nhid=512, levels=8, kernel_size=2, dropout=0, n_max=50):
        super(WordTCN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.output_size = output_size
        channel_sizes = [nhid] * levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_max = n_max
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.W_out = nn.Linear(in_features=output_size, out_features=vocab_size, bias=True)
        self.tcn = TCN(input_size=input_channel, output_size=output_size, num_channels=channel_sizes, kernel_size=kernel_size, dropout=dropout)

    def forward(self, topic_vec, captions) ->object:
        """

        :rtype: object
        """
        captions = self.embed(captions)
        embeddings = torch.cat([topic_vec, captions], dim=1)
        output = self.tcn.forward(embeddings)
        words = self.W_out(output)
        return words


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CheXNetDenseNet121,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseNet161,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseNet169,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseNet201,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EncoderCNN,
     lambda: ([], {'embed_size': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MultiLabelSoftmaxRegressionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet101,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet152,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (TCN,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TemporalConvNet,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VisualFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (WARPLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_ZexinYan_Medical_Report_Generation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

