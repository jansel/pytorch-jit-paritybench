import sys
_module = sys.modules[__name__]
del sys
__init__ = _module
simple_regression = _module
iris_multi_classification = _module
mnist_dnn = _module
dnn_mnist = _module
optim = _module
optim = _module
mycnn = _module
predict_cnn = _module
autoencoder = _module
denoise_autoencoder = _module
word_embeddings = _module
evaluate_cmn_eng = _module
logger = _module
model = _module
process = _module
seq2seq = _module
train = _module
convert = _module
preprare_data = _module
speech_command = _module
make_dataset = _module
model = _module
run = _module
speech_loader = _module
train = _module
text_classification = _module
model = _module
mydatasets = _module
text_classification = _module

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


from itertools import count


import numpy as np


import torch


import torch.autograd


import torch.nn.functional as F


from torch.autograd import Variable


from torch.optim import SGD


import torch.nn as nn


import torch.utils.data as Data


from torch import nn


from torch import optim


from torch.utils.data import DataLoader


import torch.utils as utils


import torch.autograd as autograd


import torch.optim as optim


from torch.nn import functional as F


from torch.utils.data import Dataset


import random


class Net(torch.nn.Module):
    """
    定义网络
    """

    def __init__(self, n_feature, n_hidden, n_output):
        """
        初始化函数，接受自定义输入特征维数，隐藏层特征维数，输出层特征维数
        """
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
        前向传播过程
        """
        x = F.sigmoid(self.hidden(x))
        x = self.predict(x)
        out = F.log_softmax(x, dim=1)
        return out


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class Cnn(nn.Module):

    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, 6, 3, stride=1, padding
            =1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5,
            stride=1, padding=0), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn
            .Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)
        out = self.fc(out)
        return out


class Cnn(nn.Module):

    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, 6, 3, stride=1, padding
            =1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5,
            stride=1, padding=0), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn
            .Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 400)
        out = self.fc(out)
        return out


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 1000), nn.ReLU(True
            ), nn.Linear(1000, 500), nn.ReLU(True), nn.Linear(500, 250), nn
            .ReLU(True), nn.Linear(250, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 250), nn.ReLU(True), nn.
            Linear(250, 500), nn.ReLU(True), nn.Linear(500, 1000), nn.ReLU(
            True), nn.Linear(1000, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU
            (), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=1), nn.
            ReLU(), nn.BatchNorm2d(32), nn.Conv2d(32, 64, 3, padding=1), nn
            .ReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.
            ReLU(), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.Conv2d(
            128, 256, 3, padding=1), nn.ReLU())


batch_size = 4


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU
            (), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=1), nn.
            ReLU(), nn.BatchNorm2d(32), nn.Conv2d(32, 64, 3, padding=1), nn
            .ReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.
            ReLU(), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.Conv2d(
            128, 256, 3, padding=1), nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 2, 1, 1
            ), nn.ReLU(), nn.BatchNorm2d(128), nn.ConvTranspose2d(128, 128,
            3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.ConvTranspose2d(
            128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.
            ConvTranspose2d(64, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 1, 1), nn
            .ReLU(), nn.BatchNorm2d(32), nn.ConvTranspose2d(32, 32, 3, 1, 1
            ), nn.ReLU(), nn.BatchNorm2d(32), nn.ConvTranspose2d(32, 1, 3, 
            2, 1, 1), nn.ReLU())

    def forward(self, x):
        out = x.view(batch_size, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    """
    编码器的定义
    """

    def __init__(self, input_size, hidden_size, n_layers=1):
        """
        初始化过程
        :param input_size: 输入向量长度，这里是词汇表大小
        :param hidden_size: 隐藏层大小
        :param n_layers: 叠加层数
        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        前向计算过程
        :param input: 输入
        :param hidden: 隐藏层状态
        :return: 编码器输出，隐藏层状态
        """
        try:
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            for i in range(self.n_layers):
                output, hidden = self.gru(output, hidden)
            return output, hidden
        except Exception as err:
            logger.error(err)

    def initHidden(self):
        """
        隐藏层状态初始化
        :return: 初始化过的隐藏层状态
        """
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result
        else:
            return result


class DecoderRNN(nn.Module):
    """
    解码器定义
    """

    def __init__(self, hidden_size, output_size, n_layers=1):
        """
        初始化过程
        :param hidden_size: 隐藏层大小
        :param output_size: 输出大小
        :param n_layers: 叠加层数
        """
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        前向计算过程
        :param input: 输入信息
        :param hidden: 隐藏层状态
        :return: 解码器输出，隐藏层状态
        """
        try:
            output = self.embedding(input).view(1, 1, -1)
            for i in range(self.n_layers):
                output = F.relu(output)
                output, hidden = self.gru(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden
        except Exception as err:
            logger.error(err)

    def initHidden(self):
        """
        隐藏层状态初始化
        :return: 初始化过的隐藏层状态
        """
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result
        else:
            return result


MAX_LENGTH = 25


class AttnDecoderRNN(nn.Module):
    """
    带注意力的解码器的定义
    """

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1,
        max_length=MAX_LENGTH):
        """
        带注意力的解码器初始化过程
        :param hidden_size: 隐藏层大小
        :param output_size: 输出大小
        :param n_layers: 叠加层数
        :param dropout_p: dropout率定义
        :param max_length: 接受的最大句子长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        """
        前向计算过程
        :param input: 输入信息
        :param hidden: 隐藏层状态
        :param encoder_output: 编码器分时刻的输出
        :param encoder_outputs: 编码器全部输出
        :return: 解码器输出，隐藏层状态，注意力权重
        """
        try:
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0],
                hidden[0]), 1)))
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                encoder_outputs.unsqueeze(0))
            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)
            for i in range(self.n_layers):
                output = F.relu(output)
                output, hidden = self.gru(output, hidden)
            output = F.log_softmax(self.out(output[0]))
            return output, hidden, attn_weights
        except Exception as err:
            logger.error(err)

    def initHidden(self):
        """
        隐藏层状态初始化
        :return: 初始化过的隐藏层状态
        """
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result
        else:
            return result


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        Ci = 1
        kernel_num = args.kernel_num
        kernel_sizes = args.kernel_sizes
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs_list = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (
            kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_list]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_xiaobaoonline_pytorch_in_action(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Encoder(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_001(self):
        self._check(NGramLanguageModeler(*[], **{'vocab_size': 4, 'embedding_dim': 4, 'context_size': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(autoencoder(*[], **{}), [torch.rand([784, 784])], {})

