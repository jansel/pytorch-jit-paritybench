import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
preprocess_images = _module
train = _module
train_blend = _module
utils = _module
config = _module
dataset = _module
train = _module
utils = _module
config = _module
dataset = _module
extract_images_from_csv = _module
train = _module
utils = _module
dataset = _module
train = _module
utils = _module
sort_w_attention = _module
utils = _module
googLeNet = _module
import_all_networks = _module
lenet = _module
resnet = _module
vgg = _module
train = _module
import_utils = _module
mnist_data = _module
utils = _module
build_vocabulary = _module
create_freq_vectors = _module
naivebayes = _module
generating_names = _module
main = _module
classification = _module
detection = _module
full_pytorch_example = _module
segmentation = _module
custom_dataset = _module
loader_customtext = _module
pytorch_bidirectional_lstm = _module
pytorch_init_weights = _module
pytorch_loadsave = _module
pytorch_lr_ratescheduler = _module
pytorch_mixed_precision_example = _module
pytorch_pretrain_finetune = _module
pytorch_progress_bar = _module
pytorch_rnn_gru_lstm = _module
pytorch_simple_CNN = _module
pytorch_simple_fullynet = _module
pytorch_std_mean = _module
pytorch_tensorbasics = _module
pytorch_tensorboard_ = _module
pytorch_transforms = _module
pytorch_set_seeds = _module
lenet5_pytorch = _module
pytorch_efficientnet = _module
pytorch_inceptionet = _module
pytorch_resnet = _module
pytorch_vgg_implementation = _module
fc_gan = _module
model = _module
train = _module
model = _module
train = _module
model = _module
train = _module
utils = _module
config = _module
dataset = _module
discriminator_model = _module
generator_model = _module
train = _module
utils = _module
config = _module
dataset = _module
loss = _module
model = _module
train = _module
utils = _module
config = _module
dataset = _module
discriminator_model = _module
generator_model = _module
train = _module
utils = _module
config = _module
model = _module
train = _module
utils = _module
config = _module
dataset = _module
loss = _module
model = _module
train = _module
utils = _module
config = _module
make_resized_data = _module
model = _module
prepare_data = _module
train = _module
utils = _module
dataset = _module
model = _module
train = _module
utils = _module
seq2seq = _module
utils = _module
seq2seq_attention = _module
utils = _module
model = _module
train = _module
get_loader = _module
model = _module
train = _module
utils = _module
nst = _module
seq2seq_transformer = _module
utils = _module
torchtext_tutorial1 = _module
torchtext_tutorial2 = _module
torchtext_tutorial3 = _module
transformer_from_scratch = _module
generate_csv = _module
dataset = _module
loss = _module
model = _module
train = _module
utils = _module
config = _module
dataset = _module
loss = _module
model = _module
train = _module
utils = _module
iou = _module
mean_avg_precision = _module
nms = _module
augmentations = _module
config = _module
dataset = _module
model = _module
train = _module
utils = _module
process_data = _module
train_isic = _module
tutorial8_keras_subclassing = _module
alexnet = _module
test = _module
block = _module
googlenet = _module
lenet5 = _module
vggnet = _module
decision_tree = _module
kmeansclustering = _module
knn = _module
linear_regression_gradient_descent = _module
linear_regression_normal_equation = _module
logistic_regression = _module
NN = _module
random_forest = _module
svm = _module
metrics = _module
LinearRegression_GD = _module
LinearRegression_normal = _module
iou_test = _module
map_test = _module
nms_test = _module

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


import torch


import pandas as pd


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch import nn


from torch import optim


from sklearn.metrics import cohen_kappa_score


from torchvision.utils import save_image


import warnings


import torch.nn.functional as F


import re


from sklearn.metrics import log_loss


import matplotlib.pyplot as plt


from torch.utils.data import TensorDataset


from torch.utils.data.dataset import random_split


from math import ceil


from sklearn import metrics


import torch.nn as nn


import torch.optim as optim


import random


from torch.utils.tensorboard import SummaryWriter


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torch.backends.cudnn as cudnn


from torch.utils.data import SubsetRandomSampler


import string


from torch.utils.data import WeightedRandomSampler


import torchvision


from torch.nn.utils.rnn import pad_sequence


import copy


import time


import torch.nn


from torchvision.models import vgg19


from math import log2


from scipy.stats import truncnorm


import torchvision.transforms.functional as TF


from torchtext.datasets import Multi30k


from torchtext.data.metrics import bleu_score


from torchvision import transforms


import torchvision.models as models


import torchvision.transforms.functional as FT


import matplotlib.patches as patches


from collections import Counter


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm1d((1536 + 1) * 2), nn.Linear((1536 + 1) * 2, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Dropout(0.2), nn.Linear(500, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2), nn.Linear(100, 2))

    def forward(self, x):
        return self.model(x)


class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        """
        Here we define the layers of the network. We create two fully connected layers

        Parameters:
            input_size: the size of the input, in this case 784 (28x28)
            num_classes: the number of classes we want to predict, in this case 10 (0-9)

        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)

        Parameters:
            x: mnist images

        Returns:
            out: the output of the network
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, 'Embedding size needs to be divisible by heads'
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        attention = torch.softmax(energy / self.embed_size ** (1 / 2), dim=3)
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size), nn.ReLU(), nn.Linear(forward_expansion * embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):

    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self, in_channels, out1x1, out3x3reduced, out3x3, out5x5reduced, out5x5, outpool):
        super().__init__()
        self.branch_1 = BasicConv2d(in_channels, out1x1, kernel_size=1, stride=1)
        self.branch_2 = nn.Sequential(BasicConv2d(in_channels, out3x3reduced, kernel_size=1), BasicConv2d(out3x3reduced, out3x3, kernel_size=3, padding=1))
        self.branch_3 = nn.Sequential(BasicConv2d(in_channels, out5x5reduced, kernel_size=1), BasicConv2d(out5x5reduced, out5x5, kernel_size=3, padding=1), BasicConv2d(out5x5, out5x5, kernel_size=3, padding=1))
        self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), BasicConv2d(in_channels, outpool, kernel_size=1))

    def forward(self, x):
        y1 = self.branch_1(x)
        y2 = self.branch_2(x)
        y3 = self.branch_3(x)
        y4 = self.branch_4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Inception_block(nn.Module):

    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(conv_block(in_channels, red_3x3, kernel_size=1), conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1))
        self.branch3 = nn.Sequential(conv_block(in_channels, red_5x5, kernel_size=1), conv_block(red_5x5, out_5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), conv_block(in_channels, out_1x1pool, kernel_size=1))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class GoogLeNet(nn.Module):

    def __init__(self, aux_logits=True, num_classes=1000):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        self.conv1 = conv_block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class residual_template(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.identity_downsample is not None:
            residual = self.identity_downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(intermediate_channels * 4))
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


sequence_length = 28


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class BRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class CNN(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)


class RNN_GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class RNN_LSTM(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded, (h0, c0))
        prediction = self.fc_out(outputs[-1, :, :])
        return prediction


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, reduced_dim, 1), nn.SiLU(), nn.Conv2d(reduced_dim, in_channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)
        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Sequential(CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim), SqueezeExcitation(hidden_dim, reduced_dim), nn.Conv2d(hidden_dim, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


base_model = [[1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5], [6, 80, 3, 2, 3], [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]]


phi_values = {'b0': (0, 224, 0.2), 'b1': (0.5, 240, 0.2), 'b2': (1, 260, 0.3), 'b3': (2, 300, 0.3), 'b4': (3, 380, 0.4), 'b5': (4, 456, 0.4), 'b6': (5, 528, 0.5), 'b7': (6, 600, 0.5)}


class EfficientNet(nn.Module):

    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes))

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            for layer in range(layers_repeats):
                features.append(InvertedResidualBlock(in_channels, out_channels, expand_ratio=expand_ratio, stride=stride if layer == 0 else 1, kernel_size=kernel_size, padding=kernel_size // 2))
                in_channels = out_channels
        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


class block(nn.Module):

    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


VGG_types = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


class VGG_net(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        self.fcs = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x


factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class Discriminator(nn.Module):

    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.final_block = nn.Sequential(WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.2), WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1), nn.LeakyReLU(0.2), WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1))

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


class WSLinear(nn.Module):

    def __init__(self, in_features, out_features, gain=2):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class InjectNoise(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


class GenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-08

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(PixelNorm(), WSLinear(z_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim), nn.ReLU(), WSLinear(w_dim, w_dim))

    def forward(self, x):
        return self.mapping(x)


class Generator(nn.Module):

    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.starting_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = InjectNoise(in_channels)
        self.initial_noise2 = InjectNoise(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])
        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)
        if steps == 0:
            return self.initial_rgb(x)
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = self.prog_blocks[step](upscaled, w)
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect') if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2))
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class ResidualBlock(nn.Module):

    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [nn.Sequential(CNNBlock(channels, channels // 2, kernel_size=1), CNNBlock(channels // 2, channels, kernel_size=3, padding=1))]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class VGGLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class UpsampleBlock(nn.Module):

    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class DenseResidualBlock(nn.Module):

    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        for i in range(5):
            self.blocks.append(ConvBlock(in_channels + channels * i, channels if i <= 3 else in_channels, kernel_size=3, stride=1, padding=1, use_act=True if i <= 3 else False))

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):

    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
        return [vocabulary.itos[idx] for idx in result_caption]


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cpu', max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-06)


class YoloLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        no_object_loss = self.bce(predictions[..., 0:1][noobj], target[..., 0:1][noobj])
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())
        return self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss


architecture_config = [(7, 64, 2, 3), 'M', (3, 192, 1, 1), 'M', (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), 'M', [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), 'M', [(1, 512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1), (3, 1024, 1, 1), (3, 1024, 1, 1)]


class Yolov1(nn.Module):

    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(), nn.Linear(1024 * S * S, 496), nn.Dropout(0.0), nn.LeakyReLU(0.1), nn.Linear(496, S * S * (C + B * 5)))


class ScalePrediction(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1), CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1))
        self.num_classes = num_classes

    def forward(self, x):
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class YOLOv3(nn.Module):

    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == 'S':
                    layers += [ResidualBlock(in_channels, use_residual=False, num_repeats=1), CNNBlock(in_channels, in_channels // 2, kernel_size=1), ScalePrediction(in_channels // 2, num_classes=self.num_classes)]
                    in_channels = in_channels // 2
                elif module == 'U':
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers


class Net(nn.Module):

    def __init__(self, net_version, num_classes):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-' + net_version)
        self.backbone._fc = nn.Sequential(nn.Linear(1280, num_classes))

    def forward(self, x):
        return self.backbone(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaIN,
     lambda: ([], {'channels': 4, 'w_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNNBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlock,
     lambda: ([], {'embed_size': 4, 'heads': 4, 'forward_expansion': 4, 'dropout': 0.5, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception,
     lambda: ([], {'in_channels': 4, 'out1x1': 4, 'out3x3reduced': 4, 'out3x3': 4, 'out5x5reduced': 4, 'out5x5': 4, 'outpool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception_block,
     lambda: ([], {'in_channels': 4, 'out_1x1': 4, 'red_3x3': 4, 'out_3x3': 4, 'red_5x5': 4, 'out_5x5': 4, 'out_1x1pool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InjectNoise,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MappingNetwork,
     lambda: ([], {'z_dim': 4, 'w_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NN,
     lambda: ([], {'input_size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalePrediction,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'embed_size': 4, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SqueezeExcitation,
     lambda: ([], {'in_channels': 4, 'reduced_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerBlock,
     lambda: ([], {'embed_size': 4, 'heads': 4, 'dropout': 0.5, 'forward_expansion': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (UNET,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGGLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (VariationalAutoEncoder,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WSConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WSLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (YOLOv3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_aladdinpersson_Machine_Learning_Collection(_paritybench_base):
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

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

