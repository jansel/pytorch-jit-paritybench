import sys
_module = sys.modules[__name__]
del sys
args = _module
data_loader = _module
image2embedding = _module
bigrams = _module
get_vocab = _module
mk_dataset = _module
params = _module
proc = _module
rank = _module
tokenize_instructions = _module
utils = _module
test = _module
train = _module
trijoint = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.utils.data as data


import numpy as np


import torch


import time


import torch.nn as nn


import torch.nn.parallel


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import random


class TableModule(nn.Module):

    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


def get_parser():
    parser = argparse.ArgumentParser(description='tri-joint parameters')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--img_path', default='data/images/')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--workers', default=30, type=int)
    parser.add_argument('--batch_size', default=160, type=int)
    parser.add_argument('--snapshots', default='snapshots/', type=str)
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--nRNNs', default=1, type=int)
    parser.add_argument('--srnnDim', default=1024, type=int)
    parser.add_argument('--irnnDim', default=300, type=int)
    parser.add_argument('--imfeatDim', default=2048, type=int)
    parser.add_argument('--stDim', default=1024, type=int)
    parser.add_argument('--ingrW2VDim', default=300, type=int)
    parser.add_argument('--maxSeqlen', default=20, type=int)
    parser.add_argument('--maxIngrs', default=20, type=int)
    parser.add_argument('--maxImgs', default=5, type=int)
    parser.add_argument('--numClasses', default=1048, type=int)
    parser.add_argument('--preModel', default='resNet50', type=str)
    parser.add_argument('--semantic_reg', default=True, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=720, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ingrW2V', default='data/vocab.bin', type=str)
    parser.add_argument('--valfreq', default=10, type=int)
    parser.add_argument('--patience', default=1, type=int)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeRecipe', default=True, type=bool)
    parser.add_argument('--cos_weight', default=0.98, type=float)
    parser.add_argument('--cls_weight', default=0.01, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--path_results', default='results/', type=str)
    parser.add_argument('--model_path', default='snapshots/model_e220_v-4.700.pth.tar', type=str)
    parser.add_argument('--test_image_path', default='chicken.jpg', type=str)
    parser.add_argument('--embtype', default='image', type=str)
    parser.add_argument('--medr', default=1000, type=int)
    parser.add_argument('--maxlen', default=20, type=int)
    parser.add_argument('--vocab', default='vocab.txt', type=str)
    parser.add_argument('--dataset', default='../data/recipe1M/', type=str)
    parser.add_argument('--sthdir', default='../data/', type=str)
    return parser


parser = get_parser()


opts = parser.parse_args()


class stRNN(nn.Module):

    def __init__(self):
        super(stRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)

    def forward(self, x, sq_lengths):
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        out, hidden = self.lstm(packed_seq)
        _, original_idx = sorted_idx.sort(0, descending=False)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
        idx = (sq_lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        output = output.view(output.size(0), output.size(1) * output.size(2))
        return output


class ingRNN(nn.Module):

    def __init__(self):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
        _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)
        self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        x = self.embs(x)
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        out, hidden = self.irnn(packed_seq)
        _, original_idx = sorted_idx.sort(0, descending=False)
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden[0])
        output = hidden[0].gather(1, unsorted_idx).transpose(0, 1).contiguous()
        output = output.view(output.size(0), output.size(1) * output.size(2))
        return output


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class im2recipe(nn.Module):

    def __init__(self):
        super(im2recipe, self).__init__()
        if opts.preModel == 'resNet50':
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.visionMLP = nn.Sequential(*modules)
            self.visual_embedding = nn.Sequential(nn.Linear(opts.imfeatDim, opts.embDim), nn.Tanh())
            self.recipe_embedding = nn.Sequential(nn.Linear(opts.irnnDim * 2 + opts.srnnDim, opts.embDim, opts.embDim), nn.Tanh())
        else:
            raise Exception('Only resNet50 model is implemented.')
        self.stRNN_ = stRNN()
        self.ingRNN_ = ingRNN()
        self.table = TableModule()
        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, y1, y2, z1, z2):
        recipe_emb = self.table([self.stRNN_(y1, y2), self.ingRNN_(z1, z2)], 1)
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)
        if opts.semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            output = [visual_emb, recipe_emb]
        return output

