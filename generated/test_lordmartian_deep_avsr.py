import sys
_module = sys.modules[__name__]
del sys
checker = _module
config = _module
lrs2_dataset = _module
utils = _module
demo = _module
audio_net = _module
lrs2_char_lm = _module
preprocess = _module
pretrain = _module
test = _module
train = _module
decoders = _module
general = _module
metrics = _module
preprocessing = _module
checker = _module
lrs2_dataset = _module
utils = _module
demo = _module
av_net = _module
lrs2_char_lm = _module
visual_frontend = _module
preprocess = _module
pretrain = _module
test = _module
train = _module
decoders = _module
general = _module
metrics = _module
preprocessing = _module
checker = _module
lrs2_dataset = _module
utils = _module
demo = _module
lrs2_char_lm = _module
video_net = _module
visual_frontend = _module
preprocess = _module
pretrain = _module
test = _module
train = _module
decoders = _module
general = _module
metrics = _module
preprocessing = _module

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


import numpy as np


import matplotlib.pyplot as plt


from scipy.io import wavfile


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


from scipy import signal


from scipy.special import softmax


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.optim as optim


from torch.utils.data import DataLoader


import matplotlib


from torch.utils.data import random_split


from itertools import groupby


class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch


class AudioNet(nn.Module):
    """
    An audio-only speech transcription model based on the Transformer architecture.
    Architecture: A stack of 12 Transformer encoder layers,
                  first 6 form the Encoder and the last 6 form the Decoder.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
           is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
        super(AudioNet, self).__init__()
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.audioDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batch = self.audioConv(inputBatch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        batch = self.positionalEncoding(batch)
        batch = self.audioEncoder(batch)
        batch = self.audioDecoder(batch)
        batch = batch.transpose(0, 1).transpose(1, 2)
        batch = self.outputConv(batch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(batch, dim=2)
        return outputBatch


class LRS2CharLM(nn.Module):
    """
    A character-level language model for the LRS2 Dataset.
    Architecture: Unidirectional 4-layered 1024-dim LSTM model
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe (''), space ( )
    Output: Log probabilities over the character set
    Note: The space character plays the role of the start-of-sequence token as well.
    """

    def __init__(self):
        super(LRS2CharLM, self).__init__()
        self.embedding = nn.Embedding(38, 1024, padding_idx=None)
        self.lstm = nn.LSTM(1024, 1024, num_layers=4)
        self.fc = nn.Linear(1024, 38)
        return

    def forward(self, inputBatch, initStateBatch):
        batch = self.embedding(inputBatch)
        if initStateBatch != None:
            batch, finalStateBatch = self.lstm(batch, initStateBatch)
        else:
            batch, finalStateBatch = self.lstm(batch)
        batch = batch.transpose(0, 1)
        outputBatch = F.log_softmax(self.fc(batch), dim=2)
        outputBatch = outputBatch.transpose(0, 1)
        return outputBatch, finalStateBatch


class AVNet(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
        super(AVNet, self).__init__()
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.jointConv = nn.Conv1d(2 * dModel, dModel, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch
        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch)
        else:
            audioBatch = None
        if videoInputBatch is not None:
            videoBatch = self.positionalEncoding(videoInputBatch)
            videoBatch = self.videoEncoder(videoBatch)
        else:
            videoBatch = None
        if audioBatch is not None and videoBatch is not None:
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif audioBatch is None and videoBatch is not None:
            jointBatch = videoBatch
        elif audioBatch is not None and videoBatch is None:
            jointBatch = audioBatch
        else:
            None
            exit()
        jointBatch = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))
        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))
        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class VisualFrontend(nn.Module):
    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(VisualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False), nn.BatchNorm3d(64, momentum=0.01, eps=0.001), nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.resnet = ResNet()
        return

    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch


class VideoNet(nn.Module):
    """
    A video-only speech transcription model based on the Transformer architecture.
    Architecture: A stack of 12 Transformer encoder layers,
                  first 6 form the Encoder and the last 6 form the Decoder.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout, numClasses):
        super(VideoNet, self).__init__()
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        batch = self.positionalEncoding(inputBatch)
        batch = self.videoEncoder(batch)
        batch = self.videoDecoder(batch)
        batch = batch.transpose(0, 1).transpose(1, 2)
        batch = self.outputConv(batch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(batch, dim=2)
        return outputBatch


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AudioNet,
     lambda: ([], {'dModel': 4, 'nHeads': 4, 'numLayers': 4, 'peMaxLen': 4, 'inSize': 4, 'fcHiddenSize': 4, 'dropout': 0.5, 'numClasses': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'dModel': 4, 'maxLen': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResNetLayer,
     lambda: ([], {'inplanes': 4, 'outplanes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VideoNet,
     lambda: ([], {'dModel': 4, 'nHeads': 4, 'numLayers': 4, 'peMaxLen': 4, 'fcHiddenSize': 4, 'dropout': 0.5, 'numClasses': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (VisualFrontend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 1, 128, 128])], {}),
     True),
]

class Test_lordmartian_deep_avsr(_paritybench_base):
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

