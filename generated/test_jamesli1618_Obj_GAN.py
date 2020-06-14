import sys
_module = sys.modules[__name__]
del sys
miscc = _module
utils = _module
prepare_bbox_label = _module
prepare_intp_data = _module
sample = _module
seq2seq = _module
dataset = _module
prepare_dataset = _module
evaluator = _module
loss = _module
loss = _module
DecoderRNN = _module
PreEncoderRNN = _module
models = _module
attention = _module
baseRNN = _module
seq2seq = _module
optim = _module
optim = _module
trainer = _module
supervised_trainer = _module
util = _module
checkpoint = _module
vis = _module
GlobalAttention = _module
evaluator = _module
main = _module
config = _module
inception_score_tf = _module
load = _module
losses = _module
utils = _module
model = _module
faster_rcnn = _module
faster_rcnn = _module
resnet = _module
vgg16 = _module
nms = _module
_ext = _module
build = _module
nms_cpu = _module
nms_gpu = _module
nms_wrapper = _module
roi_align = _module
functions = _module
modules = _module
roi_align = _module
roi_crop = _module
crop_resize = _module
gridgen = _module
gridgen = _module
roi_crop = _module
roi_pooling = _module
roi_pool = _module
roi_pool = _module
rpn = _module
anchor_target_layer = _module
bbox_transform = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer_cascade = _module
rpn = _module
blob = _module
logger = _module
net_utils = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
roi_data_layer = _module
minibatch = _module
roibatchLoader = _module
roidb = _module
setup = _module
testDataset = _module
trainDataset = _module
trainer = _module
datasets = _module
losses = _module
utils = _module
model = _module
trainer = _module

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


import numpy as np


from torch.nn import init


import torch


import torch.nn as nn


from torch.autograd import Variable


from copy import deepcopy


import math


import random


from random import randint


import torch.nn.functional as F


import torch.distributions as distributions


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import itertools


import torch.backends.cudnn as cudnn


from collections import defaultdict


import numpy.random as random


from scipy import linalg


import torch.nn.parallel


import torch.utils.model_zoo as model_zoo


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import numpy.random as npr


import torch.optim as optim


class PreEncoderRNN(nn.Module):

    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128,
        nlayers=1, bidirectional=True):
        super(PreEncoderRNN, self).__init__()
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers,
                batch_first=True, dropout=self.drop_prob, bidirectional=
                self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden, self.nlayers,
                batch_first=True, dropout=self.drop_prob, bidirectional=
                self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers * self.num_directions,
                bsz, self.nhidden).zero_()), Variable(weight.new(self.
                nlayers * self.num_directions, bsz, self.nhidden).zero_())
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            captions (batch, seq_len): tensor containing the features of the input sequence.
            cap_lens (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size = captions.size(0)
        hidden = self.init_hidden(batch_size)
        emb = self.drop(self.encoder(captions))
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        return output, hidden


class Attention(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        this_batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))
        if torch.cuda.is_available():
            attn_energies = attn_energies
        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, (b)],
                    encoder_outputs[b, i].unsqueeze(0))
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = energy.squeeze(0)
            energy = self.v.dot(energy)
            return energy


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = 'MASK'
    SYM_EOS = 'EOS'

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p,
        dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: {0}'.format(rnn_cell))
        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None,
        target_l_variables=None, target_x_variables=None,
        target_y_variables=None, target_w_variables=None,
        target_h_variables=None, is_training=0, early_stop_len=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable,
            input_lengths)
        result = self.decoder(encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs, target_l_variables=
            target_l_variables, target_x_variables=target_x_variables,
            target_y_variables=target_y_variables, target_w_variables=
            target_w_variables, target_h_variables=target_h_variables,
            is_training=is_training, early_stop_len=early_stop_len)
        return result


def conv1x1(in_planes, out_planes, bias=False):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
        padding=0, bias=bias)


class GlobalAttentionGeneral(nn.Module):

    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=-1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn


_global_config['TRAIN'] = 4


class GlobalBUAttentionGeneral(nn.Module):

    def __init__(self, idf, cdf):
        super(GlobalBUAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=-1)
        self.mask = None
        self.eps = 1e-08

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context1, context2):
        """
            input: batch x idf2 x ih x iw (queryL=ihxiw), label features
            context1: batch x idf2 x sourceL, glove_word_embs
            context2: batch x cdf x sourceL, word_embs
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context2.size(0), context2.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context2.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        attn = torch.bmm(targetT, context1)
        if cfg.TRAIN.BUATTN_NORM:
            norm_targetT = torch.norm(targetT, 2, dim=2, keepdim=True)
            norm_context1 = torch.norm(context1, 2, dim=1, keepdim=True)
            attn = attn / (norm_targetT * norm_context1).clamp(min=self.eps)
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class HmapResBlock(nn.Module):

    def __init__(self, channel_num):
        super(HmapResBlock, self).__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(
            channel_num, channel_num * 2, kernel_size=3, stride=1, padding=
            0, bias=False), nn.InstanceNorm2d(channel_num * 2), GLU(), nn.
            ReflectionPad2d(1), nn.Conv2d(channel_num, channel_num,
            kernel_size=3, stride=1, padding=0, bias=False), nn.
            InstanceNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


_global_config['TEXT'] = 4


_global_config['RNN_TYPE'] = 4


_global_config['CUDA'] = 4


class RNN_ENCODER(nn.Module):

    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128,
        nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers,
                batch_first=True, dropout=self.drop_prob, bidirectional=
                self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden, self.nlayers,
                batch_first=True, dropout=self.drop_prob, bidirectional=
                self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers * self.num_directions,
                bsz, self.nhidden).zero_()), Variable(weight.new(self.
                nlayers * self.num_directions, bsz, self.nhidden).zero_())
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                bsz, self.nhidden).zero_())

    def post_process_words(self, words_emb, max_len):
        batch_size, cur_len = words_emb.size(0), words_emb.size(2)
        new_words_emb = Variable(torch.zeros(batch_size, self.nhidden *
            self.num_directions, max_len))
        if cfg.CUDA:
            new_words_emb = new_words_emb
        new_words_emb[:, :, :cur_len] = words_emb
        return new_words_emb

    def forward(self, captions, cap_lens, max_len, mask=None):
        batch_size = captions.size(0)
        hidden = self.init_hidden(batch_size)
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        words_emb = self.post_process_words(words_emb, max_len)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):

    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256
        model = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth',
            'inception_v3_google-1a9a5a14.pth')
        state_dict = torch.load(model_path, map_location=lambda storage,
            loc: storage)
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        None
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        x = F.interpolate(x, size=(299, 299), mode='bilinear',
            align_corners=True)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        features = x
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        cnn_code = self.emb_cnn_code(x)
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class INCEPTION_V3(nn.Module):

    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth',
            'inception_v3_google-1a9a5a14.pth')
        state_dict = torch.load(model_path, map_location=lambda storage,
            loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input):
        x = input * 0.5 + 0.5
        x[:, (0)] = (x[:, (0)] - 0.485) / 0.229
        x[:, (1)] = (x[:, (1)] - 0.456) / 0.224
        x[:, (2)] = (x[:, (2)] - 0.406) / 0.225
        x = F.interpolate(x, size=(299, 299), mode='bilinear',
            align_corners=True)
        x = self.model(x)
        x = nn.Softmax(dim=-1)(x)
        return x


class INCEPTION_V3_FID(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        """
        super(INCEPTION_V3_FID, self).__init__()
        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth',
            'inception_v3_google-1a9a5a14.pth')
        state_dict = torch.load(model_path, map_location=lambda storage,
            loc: storage)
        inception.load_state_dict(state_dict)
        for param in inception.parameters():
            param.requires_grad = False
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.
                MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.
                Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception
                .Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.
                Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')
        x = x.clone()
        x = x * 0.5 + 0.5
        x[:, (0)] = x[:, (0)] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, (1)] = x[:, (1)] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, (2)] = x[:, (2)] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


_global_config['GAN'] = 4


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


def conv3x3(in_planes, out_planes, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
        padding=padding, bias=False)


def upBlock(in_planes, out_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2), norm(out_planes * 2), GLU())
    return block


class INIT_STAGE_G(nn.Module):

    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(nn.Linear(nz, ngf * 8 * 8 * 2, bias=False),
            nn.BatchNorm1d(ngf * 8 * 8 * 2), GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/4 x 32 x 32
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 8, 8)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        return out_code


def pprocess_bt_attns(fmaps, ih, iw, bt_mask):
    batch_size, num, max_num_rois = fmaps.size(0), fmaps.size(1), fmaps.size(2)
    fmaps = fmaps.repeat(1, 1, 1, ih * iw).view(batch_size, -1,
        max_num_rois, ih, iw)
    fmaps = fmaps.transpose(1, 2)
    fmaps = fmaps * bt_mask
    fmaps = torch.max(fmaps, dim=1)[0]
    return fmaps


class INIT_STAGE_G_MAIN(nn.Module):

    def __init__(self, ngf, nef, nef2):
        super(INIT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.ef_dim2 = nef2
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.GLB_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        nef = self.ef_dim
        nef2 = self.ef_dim2
        self.bt_att = BT_ATT_NET(ngf, nef)
        self.residual = self._make_layer(HmapResBlock, ngf * 3 + nef2)
        self.upsample = upBlock(ngf * 3 + nef2, ngf)

    def forward(self, h_code_hmap, h_code1_sent, c_code, word_embs,
        glove_word_embs, slabels_feat, mask, rois, num_rois, bt_mask,
        glb_max_num_roi):
        idf, ih, iw = h_code_hmap.size(1), h_code_hmap.size(2
            ), h_code_hmap.size(3)
        num_rois = num_rois.data.cpu().numpy().tolist()
        max_num_roi = np.amax(num_rois)
        slabels_feat = slabels_feat[:, :, :max_num_roi]
        if max_num_roi > 0:
            self.bt_att.applyMask(mask)
            bt_c_code, bt_att = self.bt_att(slabels_feat, glove_word_embs,
                word_embs)
            bt_mask = bt_mask[:, :max_num_roi]
            bt_code_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_c_code.size
                (1), 1, 1)
            bt_c_code = pprocess_bt_attns(bt_c_code, ih, iw, bt_code_mask)
            bt_att_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_att.size(1),
                1, 1)
            bt_att = pprocess_bt_attns(bt_att, ih, iw, bt_att_mask)
            bt_slabels_mask = bt_mask.unsqueeze(2).repeat(1, 1, self.
                ef_dim2, 1, 1)
            bt_slabels_code = pprocess_bt_attns(slabels_feat, ih, iw,
                bt_slabels_mask)
        else:
            bt_c_code = Variable(torch.Tensor(c_code.size()).zero_())
            bt_att = Variable(torch.Tensor(att.size()).zero_())
            bt_slabels_code = Variable(torch.Tensor(c_code.size(0), self.
                ef_dim2, c_code.size(2), c_code.size(3)).zero_())
            if cfg.CUDA:
                bt_c_code = bt_c_code
                bt_att = bt_att
                bt_slabels_code = bt_slabels_code
        out_code = torch.cat((h_code_hmap, h_code1_sent, bt_c_code,
            bt_slabels_code), 1)
        out_code = self.residual(out_code)
        out_code = self.upsample(out_code)
        return out_code


def downBlock_G(in_planes, out_planes, kernel_size=3, stride=2, padding=1,
    norm=None):
    sequence = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=False)]
    if norm is not None:
        sequence += [norm(out_planes)]
    sequence += [nn.LeakyReLU(0.2, inplace=True)]
    block = nn.Sequential(*sequence)
    return block


class G_HMAP(nn.Module):

    def __init__(self, ngf, ncf):
        super(G_HMAP, self).__init__()
        self.gf_dim = ngf
        self.in_dim = ncf
        self.define_module()

    def define_module(self):
        ncf, ngf = self.in_dim, self.gf_dim
        self.conv3x3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.downsample1 = downBlock_G(ngf, ngf * 2)

    def forward(self, hmap):
        """
        :param hmap: batch x ncf x hmap_size x hmap_size
        :return: batch x ngf*2 x hmap_size//2 x hmap_size//2
        """
        out_code = self.conv3x3(hmap)
        out_code = self.downsample1(out_code)
        return out_code


class NEXT_STAGE_G_MAIN(nn.Module):

    def __init__(self, ngf, nef, nef2):
        super(NEXT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.ef_dim2 = nef2
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.LOCAL_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        nef = self.ef_dim
        nef2 = self.ef_dim2
        self.att = ATT_NET(ngf, nef)
        self.bt_att = BT_ATT_NET(ngf, nef)
        self.residual = self._make_layer(HmapResBlock, ngf * 3 + nef2)
        self.upsample = upBlock(ngf * 3 + nef2, ngf)

    def forward(self, h_code, h_code_hmap, c_code, word_embs,
        glove_word_embs, slabels_feat, mask, rois, num_rois, bt_mask,
        glb_max_num_roi):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            glove_word_embs: batch x cdf2 x sourceL (sourceL=seq_len)
            slabels_feat: batch x cdf2 x max_num_roi x 1
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        idf, ih, iw = h_code.size(1), h_code.size(2), h_code.size(3)
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        num_rois = num_rois.data.cpu().numpy().tolist()
        max_num_roi = np.amax(num_rois)
        slabels_feat = slabels_feat[:, :, :max_num_roi]
        raw_bt_c_code = Variable(torch.Tensor(c_code.size(0), idf,
            glb_max_num_roi, 1).zero_())
        if cfg.CUDA:
            raw_bt_c_code = raw_bt_c_code
        if max_num_roi > 0:
            self.bt_att.applyMask(mask)
            bt_c_code, bt_att = self.bt_att(slabels_feat, glove_word_embs,
                word_embs)
            raw_bt_c_code[:, :, :max_num_roi] = bt_c_code
            bt_mask = bt_mask[:, :max_num_roi]
            bt_code_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_c_code.size
                (1), 1, 1)
            bt_c_code = pprocess_bt_attns(bt_c_code, ih, iw, bt_code_mask)
            bt_att_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_att.size(1),
                1, 1)
            bt_att = pprocess_bt_attns(bt_att, ih, iw, bt_att_mask)
            bt_slabels_mask = bt_mask.unsqueeze(2).repeat(1, 1, self.
                ef_dim2, 1, 1)
            bt_slabels_code = pprocess_bt_attns(slabels_feat, ih, iw,
                bt_slabels_mask)
        else:
            bt_c_code = Variable(torch.Tensor(c_code.size()).zero_())
            bt_att = Variable(torch.Tensor(att.size()).zero_())
            bt_slabels_code = Variable(torch.Tensor(c_code.size(0), self.
                ef_dim2, c_code.size(2), c_code.size(3)).zero_())
            if cfg.CUDA:
                bt_c_code = bt_c_code
                bt_att = bt_att
                bt_slabels_code = bt_slabels_code
        h_c_code = torch.cat((h_code + h_code_hmap, c_code, bt_c_code,
            bt_slabels_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)
        raw_bt_c_code = raw_bt_c_code.transpose(1, 2).squeeze(-1)
        return out_code, raw_bt_c_code, att, bt_att


class GET_IMAGE_G(nn.Module):

    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


_global_config['TREE'] = 4


class G_NET(nn.Module):

    def __init__(self, num_classes):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        nef2 = cfg.TEXT.GLOVE_EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        self.num_classes = num_classes
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1_sent = INIT_STAGE_G(ngf * 4, ncf)
            self.h_net1_hmap = G_HMAP(ngf // 2, num_classes)
            self.h_net1_main = INIT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net1 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2_hmap = G_HMAP(ngf // 2, num_classes)
            self.h_net2_main = NEXT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3_hmap = G_HMAP(ngf // 2, num_classes)
            self.h_net3_main = NEXT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, glove_word_embs,
        slabels_feat, mask, hmaps, rois, fm_rois, num_rois, bt_masks,
        fm_bt_masks, glb_max_num_roi):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param glove_word_embs: batch x cdf2 x seq_len
            :param slabels_feat: batch x cdf2 x max_num_roi x 1
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs, bt_c_codes, att_maps, bt_att_maps = [], [], [], []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1_hmap = self.h_net1_hmap(hmaps[0])
            h_code1_sent = self.h_net1_sent(z_code, c_code)
            h_code1 = self.h_net1_main(h_code1_hmap, h_code1_sent, c_code,
                word_embs, glove_word_embs, slabels_feat, mask, fm_rois,
                num_rois, fm_bt_masks, glb_max_num_roi)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2_hmap = self.h_net2_hmap(hmaps[1])
            h_code2, bt_c_code2, att1, bt_att1 = self.h_net2_main(h_code1,
                h_code2_hmap, c_code, word_embs, glove_word_embs,
                slabels_feat, mask, rois[0], num_rois, bt_masks[0],
                glb_max_num_roi)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            bt_c_codes.append(bt_c_code2)
            if att1 is not None:
                att_maps.append(att1)
            if bt_att1 is not None:
                bt_att_maps.append(bt_att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3_hmap = self.h_net3_hmap(hmaps[2])
            h_code3, bt_c_code3, att2, bt_att2 = self.h_net3_main(h_code2,
                h_code3_hmap, c_code, word_embs, glove_word_embs,
                slabels_feat, mask, rois[1], num_rois, bt_masks[1],
                glb_max_num_roi)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            bt_c_codes.append(bt_c_code3)
            if att2 is not None:
                att_maps.append(att2)
            if bt_att2 is not None:
                bt_att_maps.append(bt_att2)
        return fake_imgs, bt_c_codes, att_maps, bt_att_maps, mu, logvar


class CLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(CLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size,
            kernel_size, padding=padding)

    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = Variable(torch.zeros(state_size)), Variable(torch.
                zeros(state_size))
        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = remember_gate * prev_cell + in_gate * cell_gate
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class ResBlock(nn.Module):

    def __init__(self, channel_num, norm=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2),
            norm(channel_num * 2), GLU(), conv3x3(channel_num, channel_num *
            2), norm(channel_num * 2), GLU(), conv3x3(channel_num, channel_num)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GET_SHAPE_G(nn.Module):

    def __init__(self, nbf):
        super(GET_SHAPE_G, self).__init__()
        self.img = nn.Sequential(conv1x1(nbf, 1), nn.Sigmoid())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


def downBlock_3x3(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
        stride=2, padding=1, bias=False), nn.InstanceNorm2d(out_planes), nn
        .LeakyReLU(0.2, inplace=True))
    return block


def Block3x3_relu(in_planes, out_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2), norm(
        out_planes * 2), GLU())
    return block


_global_config['ROI'] = 4


class SHP_G_NET(nn.Module):

    def __init__(self, num_classes):
        super(SHP_G_NET, self).__init__()
        self.nbf = num_classes
        self.fm_size = cfg.ROI.FM_SIZE
        self.downsample1 = downBlock_3x3(self.nbf, self.nbf * 2)
        self.downsample2 = downBlock_3x3(self.nbf * 2, self.nbf * 4)
        self.fwd_convlstm = CLSTMCell(self.nbf * 4, self.nbf * 2, 3, 1)
        self.bwd_convlstm = CLSTMCell(self.nbf * 4, self.nbf * 2, 3, 1)
        self.jointConv = Block3x3_relu(self.nbf * 8, self.nbf * 4, norm=nn.
            InstanceNorm2d)
        self.residual = self._make_layer(ResBlock, self.nbf * 4)
        self.upsample1 = upBlock(self.nbf * 4, self.nbf * 2, norm=nn.
            InstanceNorm2d)
        self.upsample2 = upBlock(self.nbf * 2, self.nbf, norm=nn.InstanceNorm2d
            )
        self.img_net = GET_SHAPE_G(self.nbf)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num, norm=nn.InstanceNorm2d))
        return nn.Sequential(*layers)

    def forward(self, z_code, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps):
        """
            :param z_code: batch x max_num_roi x self.nbf
            :param bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
            :return: 
                   fake_hmaps: batch x max_num_roi x 3 x hmap_size x hmap_size
        """
        batch_size, max_num_roi = z_code.size(0), z_code.size(1)
        z_code = z_code.unsqueeze(3)
        z_code = z_code.repeat(1, 1, 1, self.fm_size ** 2)
        z_code = z_code.view(batch_size, max_num_roi, -1, self.fm_size,
            self.fm_size)
        hmap_size = bbox_maps_fwd.size(3)
        bbox_maps_fwd = bbox_maps_fwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_fwd = self.downsample1(bbox_maps_fwd)
        h_code_fwd = self.downsample2(h_code_fwd)
        h_code_fwd = h_code_fwd.view(batch_size, max_num_roi, -1, self.
            fm_size, self.fm_size)
        bbox_maps_bwd = bbox_maps_bwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_bwd = self.downsample1(bbox_maps_bwd)
        h_code_bwd = self.downsample2(h_code_bwd)
        h_code_bwd = h_code_bwd.view(batch_size, max_num_roi, -1, self.
            fm_size, self.fm_size)
        state_fwd, state_bwd = None, None
        state_fwd_lst, state_bwd_lst = [], []
        for t in range(0, max_num_roi):
            state_fwd = self.fwd_convlstm(h_code_fwd[:, (t)], state_fwd)
            state_fwd_lst.append(state_fwd[0].unsqueeze(1))
            state_bwd = self.bwd_convlstm(h_code_bwd[:, (t)], state_bwd)
            state_bwd_lst.append(state_bwd[0].unsqueeze(1))
        h_code = []
        for t in range(0, max_num_roi):
            h_code.append(torch.cat((state_fwd_lst[t], state_bwd_lst[
                max_num_roi - t - 1], z_code[:, t:t + 1]), 2))
        h_code = torch.cat(h_code, 1)
        h_code = h_code.view(batch_size * max_num_roi, -1, self.fm_size,
            self.fm_size)
        h_code = self.jointConv(h_code)
        bbox_fmaps = bbox_fmaps.unsqueeze(2).repeat(1, 1, self.nbf * 4, 1, 1
            ).view(batch_size * max_num_roi, -1, self.fm_size, self.fm_size)
        h_code = h_code * bbox_fmaps
        h_code = self.residual(h_code)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        fake_hmaps = self.img_net(h_code)
        fake_hmaps = fake_hmaps.view(batch_size, max_num_roi, -1, hmap_size,
            hmap_size)
        return fake_hmaps


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes), nn.BatchNorm2d(
        out_planes), nn.LeakyReLU(0.2, inplace=True))
    return block


class D_GET_LOGITS(nn.Module):

    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.layer_num = cfg.GAN.LAYER_D_NUM
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * pow(2, self.layer_num -
                1) + nef, ndf * pow(2, self.layer_num - 1))
        self.outlogits = nn.Sequential(nn.Conv2d(ndf * pow(2, self.
            layer_num - 1), 1, kernel_size=4, stride=2), nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
            h_c_code = torch.cat((h_code, c_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output


def encode_image_by_ntimes(ngf, ndf, n_layer):
    sequence = [nn.Conv2d(3 + ngf, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(
        0.2, inplace=True)]
    for n in range(1, n_layer):
        nf_mult_prev = ndf * min(2 ** (n - 1), 8)
        nf_mult = ndf * min(2 ** n, 8)
        sequence += [nn.Conv2d(nf_mult_prev, nf_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf_mult), nn.LeakyReLU(0.2, inplace=True)]
    encode_img = nn.Sequential(*sequence)
    return encode_img


class PAT_D_NET64(nn.Module):

    def __init__(self, b_jcu=True):
        super(PAT_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code(x_var)
        return x_code4


class PAT_D_NET128(nn.Module):

    def __init__(self, b_jcu=True):
        super(PAT_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code(x_var)
        return x_code8


class PAT_D_NET256(nn.Module):

    def __init__(self, b_jcu=True):
        super(PAT_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code(x_var)
        return x_code16


class SHP_D_NET64(nn.Module):

    def __init__(self, num_classes):
        super(SHP_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code4 = self.img_code(x_s_var)
        return x_code4


class SHP_D_NET128(nn.Module):

    def __init__(self, num_classes):
        super(SHP_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code8 = self.img_code(x_s_var)
        return x_code8


class SHP_D_NET256(nn.Module):

    def __init__(self, num_classes):
        super(SHP_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code16 = self.img_code(x_s_var)
        return x_code16


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    levels = np.arange(im_rois.shape[0] // cfg.ROI.BOXES_NUM).astype(np.int)
    levels = np.expand_dims(levels, axis=1)
    levels = np.repeat(levels, cfg.ROI.BOXES_NUM, axis=1)
    levels = np.reshape(levels, (levels.shape[0] * levels.shape[1], 1))
    rois = im_rois * scales[levels]
    return rois, levels


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


class OBJ_SS_D_NET(nn.Module):

    def __init__(self, num_classes, b_jcu=True):
        super(OBJ_SS_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.GLOVE_EMBEDDING_DIM + cfg.GAN.GF_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.roi_size = cfg.ROI.ROI_BASE_SIZE
        self.im_scales = np.array([1])
        n_layer = 3
        self.img_code = encode_image_by_ntimes(ngf, ndf, n_layer)
        self.shp_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.roi_code = nn.Sequential(nn.Conv2d(ndf * min(2 ** (n_layer - 1
            ), 8), ndf * 4, kernel_size=4, stride=1, padding=1), nn.
            LeakyReLU(0.2, True))
        self.RoIAlignAvg = RoIAlignAvg(self.roi_size, self.roi_size, 1.0 / 16.0
            )
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf // 2, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf // 2, nef, bcondition=True)

    def forward(self, x_var, s_var, fm_rois, num_rois, img_size=512):
        fm_rois_roi = fm_rois.data.cpu().numpy()
        fm_rois_roi[:, :, ([2, 3])] = fm_rois_roi[:, :, ([0, 1])
            ] + fm_rois_roi[:, :, ([2, 3])]
        num_rois = num_rois.data.cpu().numpy().tolist()
        x_var = F.interpolate(x_var, size=(img_size, img_size), mode=
            'bilinear', align_corners=True)
        s_var = F.interpolate(s_var, size=(img_size, img_size), mode=
            'bilinear', align_corners=True)
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code64 = self.img_code(x_s_var)
        max_num_roi = np.amax(num_rois)
        batch_size = fm_rois_roi.shape[0]
        fm_rois_roi = np.reshape(fm_rois_roi, (fm_rois_roi.shape[0] *
            fm_rois_roi.shape[1], fm_rois_roi.shape[2]))
        roi_data = torch.FloatTensor(1)
        if cfg.CUDA:
            roi_data = roi_data
        vroi_data = Variable(roi_data, requires_grad=False)
        im_scales = np.array([1] * batch_size * cfg.ROI.BOXES_NUM)
        gt_rois_np = _get_rois_blob(fm_rois_roi[:, :4], im_scales)[(np.
            newaxis), :]
        gt_rois_pt = torch.from_numpy(gt_rois_np[(np.newaxis), :])
        vroi_data.data.resize_(gt_rois_pt.size()).copy_(gt_rois_pt)
        pooled_feat = self.RoIAlignAvg(x_code64, vroi_data.view(-1, 5))
        pooled_feat = self.roi_code(pooled_feat)
        pooled_feat = pooled_feat.view(batch_size, cfg.ROI.BOXES_NUM,
            pooled_feat.size(1), pooled_feat.size(2), pooled_feat.size(3))
        return pooled_feat


class OBJ_LS_D_NET(nn.Module):

    def __init__(self, num_classes, b_jcu=True):
        super(OBJ_LS_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.GLOVE_EMBEDDING_DIM + cfg.GAN.GF_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.roi_size = cfg.ROI.ROI_BASE_SIZE
        self.im_scales = np.array([1])
        n_layer = 4
        self.img_code = encode_image_by_ntimes(ngf, ndf, n_layer)
        self.shp_code = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ncf,
            ngf, kernel_size=3, padding=0), nn.InstanceNorm2d(ngf), nn.
            LeakyReLU(0.2, inplace=True))
        self.roi_code = nn.Sequential(nn.Conv2d(ndf * min(2 ** (n_layer - 1
            ), 8), ndf * 4, kernel_size=4, stride=1, padding=1), nn.
            LeakyReLU(0.2, True))
        self.RoIAlignAvg = RoIAlignAvg(self.roi_size, self.roi_size, 1.0 / 16.0
            )
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf // 2, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf // 2, nef, bcondition=True)

    def forward(self, x_var, s_var, fm_rois, num_rois, img_size=512):
        fm_rois_roi = fm_rois.data.cpu().numpy()
        fm_rois_roi[:, :, ([2, 3])] = fm_rois_roi[:, :, ([0, 1])
            ] + fm_rois_roi[:, :, ([2, 3])]
        num_rois = num_rois.data.cpu().numpy().tolist()
        x_var = F.interpolate(x_var, size=(img_size, img_size), mode=
            'bilinear', align_corners=True)
        s_var = F.interpolate(s_var, size=(img_size, img_size), mode=
            'bilinear', align_corners=True)
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code32 = self.img_code(x_s_var)
        max_num_roi = np.amax(num_rois)
        batch_size = fm_rois_roi.shape[0]
        fm_rois_roi = np.reshape(fm_rois_roi, (fm_rois_roi.shape[0] *
            fm_rois_roi.shape[1], fm_rois_roi.shape[2]))
        roi_data = torch.FloatTensor(1)
        if cfg.CUDA:
            roi_data = roi_data
        vroi_data = Variable(roi_data, requires_grad=False)
        im_scales = np.array([1] * batch_size * cfg.ROI.BOXES_NUM)
        gt_rois_np = _get_rois_blob(fm_rois_roi[:, :4], im_scales)[(np.
            newaxis), :]
        gt_rois_pt = torch.from_numpy(gt_rois_np[(np.newaxis), :])
        vroi_data.data.resize_(gt_rois_pt.size()).copy_(gt_rois_pt)
        pooled_feat = self.RoIAlignAvg(x_code32, vroi_data.view(-1, 5))
        pooled_feat = self.roi_code(pooled_feat)
        pooled_feat = pooled_feat.view(batch_size, cfg.ROI.BOXES_NUM,
            pooled_feat.size(1), pooled_feat.size(2), pooled_feat.size(3))
        return pooled_feat


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
    bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.0
        ) * smoothL1_sign + (abs_in_box_diff - 0.5 / sigma_2) * (1.0 -
        smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0
    height = input_size[0]
    width = input_size[1]
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([(x2 - x1) / (width - 1), zero, (x1 + x2 - width + 1) /
        (width - 1), zero, (y2 - y1) / (height - 1), (y1 + y2 - height + 1) /
        (height - 1)], 1).view(-1, 2, 3)
    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size,
        grid_size)))
    return grid


_global_config['POOLING_SIZE'] = 4


_global_config['POOLING_MODE'] = 4


_global_config['CROP_RESIZE_WITH_MAX_POOL'] = 4


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
            1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.
            POOLING_SIZE, 1.0 / 16.0)
        self.grid_size = (cfg.POOLING_SIZE * 2 if cfg.
            CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE)
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        base_feat = self.RCNN_base(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat,
            im_info, gt_boxes, num_boxes)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            (rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws
                ) = roi_data
            None
            None
            None
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1,
                rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1,
                rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)
        None
        None
        None
        None
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2
                :], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, (1)], grid_xy.data
                [:, :, :, (0)]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).
                detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            None
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)
        None
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(
                bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.
                view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)
                )
            bbox_pred = bbox_pred_select.squeeze(1)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        None
        None
        sys.exit()
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                rois_inside_ws, rois_outside_ws)
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return (rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox,
            RCNN_loss_cls, RCNN_loss_bbox, rois_label)

    def _init_weights(self):

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height,
            self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.
            aligned_width, self.spatial_scale, grad_output, self.rois,
            grad_input)
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
            self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new(torch.Size([input1.size(0)]) + self.grid.size()
            ).zero_()
        self.batchgrid = input1.new(torch.Size([input1.size(0)]) + self.
            grid.size()).zero_()
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.astype(self.batchgrid[i])
        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height * self.
                width, 3), torch.transpose(input1, 1, 2)).view(-1, self.
                height, self.width, 2)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(
            grad_output.view(-1, self.height * self.width, 2), 1, 2), self.
            batchgrid.view(-1, self.height * self.width, 3))
        return grad_input1


class _AffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(_AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        return self.f(input)


class CylinderGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1, 1)


class AffineGridGenV2(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width,
            3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.
            width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.
            grid.size())
        for i in range(input.size(0)):
            self.batchgrid[(i), :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :,
            :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (
            np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.mul(self.batchgrid, input1[:, :, :, 0:3])
        y = torch.mul(self.batchgrid, input1[:, :, :, 3:6])
        output = torch.cat([torch.sum(x, 3), torch.sum(y, 3)], 3)
        return output


class DenseAffine3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class DenseAffine3DGridGen_rotate(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen_with_mask(Module):

    def __init__(self, height, width, lr=1, aux_loss=False, ray_tracing=False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.
            width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                cuda.FloatTensor) * (y.ge(0).type(torch.cuda.FloatTensor) -
                y.lt(0).type(torch.cuda.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).
                type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


with_cuda = False


headers = []


sources = []


defines = []


extra_objects = ['src/nms_cuda_kernel.cu.o']


class RoIPoolFunction(Function):

    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height,
            ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height,
            ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.
                pooled_width, ctx.spatial_scale, _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.
                pooled_width, ctx.spatial_scale, features, rois, output,
                ctx.argmax)
        return output

    def backward(ctx, grad_output):
        assert ctx.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.
            pooled_width, ctx.spatial_scale, grad_output, ctx.rois,
            grad_input, ctx.argmax)
        return grad_input, None


class _RoIPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.
            spatial_scale)(features, rois)


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, (2)] - gt_boxes[:, :, (0)] + 1
        gt_boxes_y = gt_boxes[:, :, (3)] - gt_boxes[:, :, (1)] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, (2)] - anchors[:, :, (0)] + 1
        anchors_boxes_y = anchors[:, :, (3)] - anchors[:, :, (1)] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
            N, K, 4)
        iw = torch.min(boxes[:, :, :, (2)], query_boxes[:, :, :, (2)]
            ) - torch.max(boxes[:, :, :, (0)], query_boxes[:, :, :, (0)]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, (3)], query_boxes[:, :, :, (3)]
            ) - torch.max(boxes[:, :, :, (1)], query_boxes[:, :, :, (1)]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(
            batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).
            expand(batch_size, N, K), -1)
    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)
        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, (2)] - gt_boxes[:, :, (0)] + 1
        gt_boxes_y = gt_boxes[:, :, (3)] - gt_boxes[:, :, (1)] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, (2)] - anchors[:, :, (0)] + 1
        anchors_boxes_y = anchors[:, :, (3)] - anchors[:, :, (1)] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
            N, K, 4)
        iw = torch.min(boxes[:, :, :, (2)], query_boxes[:, :, :, (2)]
            ) - torch.max(boxes[:, :, :, (0)], query_boxes[:, :, :, (0)]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, (3)], query_boxes[:, :, :, (3)]
            ) - torch.max(boxes[:, :, :, (1)], query_boxes[:, :, :, (1)]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(
            batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).
            expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')
    return overlaps


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, (np.newaxis)]
    hs = hs[:, (np.newaxis)]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), 
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.
    arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[(i), :], scales) for i in
        xrange(ratio_anchors.shape[0])])
    return anchors


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, (inds)] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill
            ).type_as(data)
        ret[:, (inds), :] = data
    return ret


def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, (2)] - ex_rois[:, (0)] + 1.0
        ex_heights = ex_rois[:, (3)] - ex_rois[:, (1)] + 1.0
        ex_ctr_x = ex_rois[:, (0)] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, (1)] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, (2)] - gt_rois[:, :, (0)] + 1.0
        gt_heights = gt_rois[:, :, (3)] - gt_rois[:, :, (1)] + 1.0
        gt_ctr_x = gt_rois[:, :, (0)] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, (1)] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)
            ) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)
            ) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1, -1).expand_as(
            gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1, -1).
            expand_as(gt_heights))
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, (2)] - ex_rois[:, :, (0)] + 1.0
        ex_heights = ex_rois[:, :, (3)] - ex_rois[:, :, (1)] + 1.0
        ex_ctr_x = ex_rois[:, :, (0)] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, (1)] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, (2)] - gt_rois[:, :, (0)] + 1.0
        gt_heights = gt_rois[:, :, (3)] - gt_rois[:, :, (1)] + 1.0
        gt_ctr_x = gt_rois[:, :, (0)] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, (1)] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)
    return targets


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(
            anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self._allowed_border = 0

    def forward(self, input):
        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        batch_size = gt_boxes.size(0)
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel
            (), shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()
        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)
        total_anchors = int(K * A)
        keep = (all_anchors[:, (0)] >= -self._allowed_border) & (all_anchors
            [:, (1)] >= -self._allowed_border) & (all_anchors[:, (2)] < 
            long(im_info[0][1]) + self._allowed_border) & (all_anchors[:, (
            3)] < long(im_info[0][0]) + self._allowed_border)
        inds_inside = torch.nonzero(keep).view(-1)
        anchors = all_anchors[(inds_inside), :]
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)
            ).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)
            ).zero_()
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-05
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1
            ).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        for i in range(batch_size):
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.
                    size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1
                )[i]
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.
                    size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(
            argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)
            [(argmax_overlaps.view(-1)), :].view(batch_size, -1, 5))
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.
                RPN_POSITIVE_WEIGHT < 1)
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1
            )
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside,
            batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors,
            inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors,
            inds_inside, batch_size, fill=0)
        outputs = []
        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2
            ).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)
        bbox_targets = bbox_targets.view(batch_size, height, width, A * 4
            ).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)
        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,
            anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size,
            height, width, 4 * A).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_inside_weights)
        bbox_outside_weights = bbox_outside_weights.view(batch_size,
            anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(
            batch_size, height, width, 4 * A).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_outside_weights)
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def nms_gpu(dets, thresh):
    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()
    nms.nms_cuda(keep, dets, num_out, thresh)
    keep = keep[:num_out[0]]
    return keep


def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, (0)]
    y1 = dets[:, (1)]
    x2 = dets[:, (2)]
    y2 = dets[:, (3)]
    scores = dets[:, (4)]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return torch.IntTensor(keep)


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    return nms_gpu(dets, thresh) if force_cpu == False else nms_cpu(dets,
        thresh)


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[(i), :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[(i), :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    return boxes


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, (2)] - boxes[:, :, (0)] + 1.0
    heights = boxes[:, :, (3)] - boxes[:, :, (1)] + 1.0
    ctr_x = boxes[:, :, (0)] + 0.5 * widths
    ctr_y = boxes[:, :, (1)] + 0.5 * heights
    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_boxes = deltas.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


_global_config['USE_GPU_NMS'] = 4


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(
            scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        batch_size = bbox_deltas.size(0)
        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel
            (), shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()
        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        proposals = clip_boxes(proposals, im_info, batch_size)
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            order_single = order[i]
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            proposals_single = proposals_single[(order_single), :]
            scores_single = scores_single[order_single].view(-1, 1)
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1
                ), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[(keep_idx_i), :]
            scores_single = scores_single[(keep_idx_i), :]
            num_proposal = proposals_single.size(0)
            output[(i), :, (0)] = i
            output[(i), :num_proposal, 1:] = proposals_single
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, (2)] - boxes[:, :, (0)] + 1
        hs = boxes[:, :, (3)] - boxes[:, :, (1)] + 1
        keep = (ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size
            .view(-1, 1).expand_as(hs))
        return keep


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.
            BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.
            BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.
            BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION *
            rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights = (self.
            _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes))
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return (rois, labels, bbox_targets, bbox_inside_weights,
            bbox_outside_weights)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data,
        labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4
            ).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        for b in range(batch_size):
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[(b), (ind), :] = bbox_target_data[(b), (ind), :]
                bbox_inside_weights[(b), (ind), :] = self.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4
        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        targets = bbox_transform_batch(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            targets = (targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)
                ) / self.BBOX_NORMALIZE_STDS.expand_as(targets)
        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = gt_boxes[:, :, (4)].contiguous().view(-1).index((offset.
            view(-1),)).view(batch_size, -1)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH
                ).view(-1)
            fg_num_rois = fg_inds.numel()
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.
                BG_THRESH_HI) & (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)
                ).view(-1)
            bg_num_rois = bg_inds.numel()
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)
                    ).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = (rois_per_image -
                    fg_rois_per_this_image)
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) *
                    bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) *
                    fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) *
                    bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError(
                    'bg_num_rois = 0 and fg_num_rois = 0, this should not happen!'
                    )
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            labels_batch[i].copy_(labels[i][keep_inds])
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[(i), :, (0)] = i
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1
            :5], gt_rois_batch[:, :, :4])
        bbox_targets, bbox_inside_weights = (self.
            _get_bbox_regression_labels_pytorch(bbox_target_data,
            labels_batch, num_classes))
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights


_global_config['FEAT_STRIDE'] = 4


_global_config['ANCHOR_SCALES'] = 4


_global_config['ANCHOR_RATIOS'] = 4


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.
            anchor_scales, self.anchor_ratios)
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.
            anchor_scales, self.anchor_ratios)
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat.size(0)
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
            im_info, cfg_key))
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes,
                im_info, num_boxes))
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1
                ).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0,
                rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data
                )
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            (rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights) = rpn_data[1:]
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred,
                rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class ResBlock(nn.Module):

    def __init__(self, channel_num, norm=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2),
            norm(channel_num * 2), GLU(), conv3x3(channel_num, channel_num *
            2), norm(channel_num * 2), GLU(), conv3x3(channel_num, channel_num)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(CLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size,
            kernel_size, padding=padding)

    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))
            if cfg.CUDA:
                prev_hidden = prev_hidden
                prev_cell = prev_cell
            prev_state = prev_hidden, prev_cell
        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = remember_gate * prev_cell + in_gate * cell_gate
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class GET_IMAGE_G(nn.Module):

    def __init__(self, nbf):
        super(GET_IMAGE_G, self).__init__()
        self.img = nn.Sequential(conv1x1(nbf, 1), nn.Sigmoid())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):

    def __init__(self, num_classes):
        super(G_NET, self).__init__()
        self.nbf = num_classes
        self.fm_size = cfg.ROI.FM_SIZE
        self.downsample1 = downBlock_3x3(self.nbf, self.nbf * 2)
        self.downsample2 = downBlock_3x3(self.nbf * 2, self.nbf * 4)
        self.fwd_convlstm = CLSTMCell(self.nbf * 4, self.nbf * 2, 3, 1)
        self.bwd_convlstm = CLSTMCell(self.nbf * 4, self.nbf * 2, 3, 1)
        self.jointConv = Block3x3_relu(self.nbf * 8, self.nbf * 4, norm=nn.
            InstanceNorm2d)
        self.residual = self._make_layer(ResBlock, self.nbf * 4)
        self.upsample1 = upBlock(self.nbf * 4, self.nbf * 2, norm=nn.
            InstanceNorm2d)
        self.upsample2 = upBlock(self.nbf * 2, self.nbf, norm=nn.InstanceNorm2d
            )
        self.img_net = GET_IMAGE_G(self.nbf)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num, norm=nn.InstanceNorm2d))
        return nn.Sequential(*layers)

    def forward(self, z_code, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps):
        """
            :param z_code: batch x max_num_roi x self.nbf
            :param bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
            :return: 
                   fake_hmaps: batch x max_num_roi x 3 x hmap_size x hmap_size
        """
        batch_size, max_num_roi = z_code.size(0), z_code.size(1)
        z_code = z_code.unsqueeze(3)
        z_code = z_code.repeat(1, 1, 1, self.fm_size ** 2)
        z_code = z_code.view(batch_size, max_num_roi, -1, self.fm_size,
            self.fm_size)
        hmap_size = bbox_maps_fwd.size(3)
        bbox_maps_fwd = bbox_maps_fwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_fwd = self.downsample1(bbox_maps_fwd)
        h_code_fwd = self.downsample2(h_code_fwd)
        h_code_fwd = h_code_fwd.view(batch_size, max_num_roi, -1, self.
            fm_size, self.fm_size)
        bbox_maps_bwd = bbox_maps_bwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_bwd = self.downsample1(bbox_maps_bwd)
        h_code_bwd = self.downsample2(h_code_bwd)
        h_code_bwd = h_code_bwd.view(batch_size, max_num_roi, -1, self.
            fm_size, self.fm_size)
        state_fwd, state_bwd = None, None
        state_fwd_lst, state_bwd_lst = [], []
        for t in range(0, max_num_roi):
            state_fwd = self.fwd_convlstm(h_code_fwd[:, (t)], state_fwd)
            state_fwd_lst.append(state_fwd[0].unsqueeze(1))
            state_bwd = self.bwd_convlstm(h_code_bwd[:, (t)], state_bwd)
            state_bwd_lst.append(state_bwd[0].unsqueeze(1))
        h_code = []
        for t in range(0, max_num_roi):
            h_code.append(torch.cat((state_fwd_lst[t], state_bwd_lst[
                max_num_roi - t - 1], z_code[:, t:t + 1]), 2))
        h_code = torch.cat(h_code, 1)
        h_code = h_code.view(batch_size * max_num_roi, -1, self.fm_size,
            self.fm_size)
        h_code = self.jointConv(h_code)
        bbox_fmaps = bbox_fmaps.unsqueeze(2).repeat(1, 1, self.nbf * 4, 1, 1
            ).view(batch_size * max_num_roi, -1, self.fm_size, self.fm_size)
        h_code = h_code * bbox_fmaps
        h_code = self.residual(h_code)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        fake_hmaps = self.img_net(h_code)
        fake_hmaps = fake_hmaps.view(batch_size, max_num_roi, -1, hmap_size,
            hmap_size)
        return fake_hmaps


class D_GET_LOGITS(nn.Module):

    def __init__(self, nbf):
        super(D_GET_LOGITS, self).__init__()
        self.outlogits = nn.Sequential(nn.Conv2d(nbf // 16, 1, kernel_size=
            4, stride=4), nn.Sigmoid())

    def forward(self, h_code):
        output = self.outlogits(h_code)
        return output


def downBlock_4x4(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=4,
        stride=2, padding=1, bias=False), nn.InstanceNorm2d(out_planes), nn
        .LeakyReLU(0.2, inplace=True))
    return block


class INS_D_NET(nn.Module):

    def __init__(self, num_classes):
        super(INS_D_NET, self).__init__()
        self.nbf = num_classes
        self.downsample1 = downBlock_4x4(self.nbf + 1, self.nbf // 2)
        self.downsample2 = downBlock_4x4(self.nbf // 2, self.nbf // 4)
        self.downsample3 = downBlock_4x4(self.nbf // 4, self.nbf // 8)
        self.downsample4 = downBlock_4x4(self.nbf // 8, self.nbf // 16)
        self.get_logits = D_GET_LOGITS(self.nbf)

    def forward(self, x_var):
        x_code32 = self.downsample1(x_var)
        x_code16 = self.downsample2(x_code32)
        x_code8 = self.downsample3(x_code16)
        x_code4 = self.downsample4(x_code8)
        return x_code4


class GLB_D_NET(nn.Module):

    def __init__(self, num_classes):
        super(GLB_D_NET, self).__init__()
        self.nbf = num_classes
        self.downsample1 = downBlock_4x4(self.nbf + 1, self.nbf // 2)
        self.downsample2 = downBlock_4x4(self.nbf // 2, self.nbf // 4)
        self.downsample3 = downBlock_4x4(self.nbf // 4, self.nbf // 8)
        self.downsample4 = downBlock_4x4(self.nbf // 8, self.nbf // 16)
        self.get_logits = D_GET_LOGITS(self.nbf)

    def forward(self, x_var):
        x_code32 = self.downsample1(x_var)
        x_code16 = self.downsample2(x_code32)
        x_code8 = self.downsample3(x_code16)
        x_code4 = self.downsample4(x_code8)
        return x_code4


feature_indices = {2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51}


classifier_indices = {1, 4, 6}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear',
            align_corners=True)
        results = []
        for index, model in enumerate(self.features):
            x = model(x)
            if index in feature_indices:
                results.append(x)
        x = x.view(x.size(0), -1)
        for index, model in enumerate(self.classifier):
            x = model(x)
            if index in classifier_indices:
                results.append(x)
        return results

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jamesli1618_Obj_GAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(D_GET_LOGITS(*[], **{'nbf': 64}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Depth3DGridGen(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Depth3DGridGen_with_mask(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(GET_IMAGE_G(*[], **{'nbf': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(GET_SHAPE_G(*[], **{'nbf': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(GLB_D_NET(*[], **{'num_classes': 64}), [torch.rand([4, 65, 64, 64])], {})

    def test_007(self):
        self._check(GLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(G_HMAP(*[], **{'ngf': 4, 'ncf': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(GlobalAttentionGeneral(*[], **{'idf': 4, 'cdf': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_010(self):
        self._check(HmapResBlock(*[], **{'channel_num': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(INS_D_NET(*[], **{'num_classes': 64}), [torch.rand([4, 65, 64, 64])], {})

    def test_012(self):
        self._check(ResBlock(*[], **{'channel_num': 4}), [torch.rand([4, 4, 4, 4])], {})

