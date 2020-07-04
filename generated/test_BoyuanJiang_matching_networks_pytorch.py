import sys
_module = sys.modules[__name__]
del sys
OmniglotBuilder = _module
gen_datatset = _module
mainOmniglot = _module
matching_networks = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import math


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True), nn.BatchNorm2d(out_channels), nn.MaxPool2d(
        kernel_size=2, stride=2), nn.Dropout(keep_prob))
    return cnn_seq


class Classifier(nn.Module):

    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0,
        image_size=28):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        return x


class AttentionalClassify(nn.Module):

    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float('inf')).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.
                unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class BidirectionalLSTM(nn.Module):

    def __init__(self, layer_size, batch_size, vector_dim, use_cuda):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.
            num_layer, hidden_size=self.hidden_size, bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self, use_cuda):
        if use_cuda:
            return Variable(torch.zeros(self.lstm.num_layers * 2, self.
                batch_size, self.lstm.hidden_size), requires_grad=False
                ), Variable(torch.zeros(self.lstm.num_layers * 2, self.
                batch_size, self.lstm.hidden_size), requires_grad=False)
        else:
            return Variable(torch.zeros(self.lstm.num_layers * 2, self.
                batch_size, self.lstm.hidden_size), requires_grad=False
                ), Variable(torch.zeros(self.lstm.num_layers * 2, self.
                batch_size, self.lstm.hidden_size), requires_grad=False)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class MatchingNetwork(nn.Module):

    def __init__(self, keep_prob, batch_size=32, num_channels=1,
        learning_rate=0.001, fce=False, num_classes_per_set=20,
        num_samples_per_class=1, image_size=28, use_cuda=True):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = Classifier(layer_size=64, num_channels=num_channels,
            keep_prob=keep_prob, image_size=image_size)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        if self.fce:
            self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.
                batch_size, vector_dim=self.g.outSize, use_cuda=use_cuda)

    def forward(self, support_set_images, support_set_y_one_hot,
        target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, (i), :, :])
            encoded_images.append(gen_encode)
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images)
        if self.fce:
            outputs = self.lstm(output)
        similarites = self.dn(support_set=output[:-1], input_image=output[-1])
        preds = self.classify(similarites, support_set_y=support_set_y_one_hot)
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())
        return accuracy, crossentropy_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_BoyuanJiang_matching_networks_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AttentionalClassify(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(Classifier(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_002(self):
        self._check(DistanceNetwork(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

