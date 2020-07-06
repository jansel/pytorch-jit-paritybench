import sys
_module = sys.modules[__name__]
del sys
datasets = _module
miniImagenetOneShot = _module
omniglot = _module
omniglotNShot = _module
OneShotBuilder = _module
OneShotMiniImageNetBuilder = _module
experiments = _module
logger = _module
mainMiniImageNet = _module
mainOmniglot = _module
AttentionalClassify = _module
BidirectionalLSTM = _module
Classifier = _module
DistanceNetwork = _module
MatchingNetwork = _module
models = _module
option = _module
utils = _module
create_miniImagenet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import math


import collections


import numpy as np


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.init as init


import torch.nn.functional as F


class AttentionalClassify(nn.Module):

    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class BidirectionalLSTM(nn.Module):

    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3 layer, 100
        :param batch_size: The experiments batch size
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)
        """
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layers, hidden_size=self.hidden_size, bidirectional=True)

    def forward(self, inputs):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param x: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        c0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False)
        h0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn


def convLayer(in_planes, out_planes, useDropout=False):
    """3x3 convolution with padding"""
    seq = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(out_planes), nn.ReLU(True), nn.MaxPool2d(kernel_size=2, stride=2))
    if useDropout:
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq


class Classifier(nn.Module):

    def __init__(self, layer_size, nClasses=0, num_channels=1, useDropout=False, image_size=28):
        super(Classifier, self).__init__()
        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        self.layer1 = convLayer(num_channels, layer_size, useDropout)
        self.layer2 = convLayer(layer_size, layer_size, useDropout)
        self.layer3 = convLayer(layer_size, layer_size, useDropout)
        self.layer4 = convLayer(layer_size, layer_size, useDropout)
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        if nClasses > 0:
            self.useClassification = True
            self.layer5 = nn.Linear(self.outSize, nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False
        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.useClassification:
            x = self.layer5(x)
        return x


class DistanceNetwork(nn.Module):

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float('inf')).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities


class MatchingNetwork(nn.Module):

    def __init__(self, keep_prob, batch_size=100, num_channels=1, learning_rate=0.001, fce=False, num_classes_per_set=5, num_samples_per_class=1, nClasses=0, image_size=28):
        super(MatchingNetwork, self).__init__()
        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param nClasses: total number of classes. It changes the output size of the classifier g with a final FC layer.
        :param image_input: size of the input image. It is needed in case we want to create the last FC classification 
        """
        self.batch_size = batch_size
        self.fce = fce
        self.g = Classifier(layer_size=64, num_channels=num_channels, nClasses=nClasses, image_size=image_size)
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size, vector_dim=self.g.outSize)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_images, support_set_labels_one_hot, target_image, target_label):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return: 
        """
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, (i), :, :, :])
            encoded_images.append(gen_encode)
        for i in np.arange(target_image.size(1)):
            gen_encode = self.g(target_image[:, (i), :, :, :])
            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)
            if self.fce:
                outputs, hn, cn = self.lstm(outputs)
            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = similarities.t()
            preds = self.classify(similarities, support_set_y=support_set_labels_one_hot)
            values, indices = preds.max(1)
            if i == 0:
                accuracy = torch.mean((indices.squeeze() == target_label[:, (i)]).float())
                crossentropy_loss = F.cross_entropy(preds, target_label[:, (i)].long())
            else:
                accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, (i)]).float())
                crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, target_label[:, (i)].long())
            encoded_images.pop()
        return accuracy / target_image.size(1), crossentropy_loss / target_image.size(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionalClassify,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BidirectionalLSTM,
     lambda: ([], {'layer_sizes': [4, 4], 'batch_size': 4, 'vector_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Classifier,
     lambda: ([], {'layer_size': 1}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (DistanceNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_gitabcworld_MatchingNetworks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

