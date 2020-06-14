import sys
_module = sys.modules[__name__]
del sys
VAD_segments = _module
data_load = _module
data_preprocess = _module
dvector_create = _module
hparam = _module
speech_embedder_net = _module
train_speech_embedder = _module
utils = _module

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


import torch.nn as nn


import random


from torch.utils.data import DataLoader


import numpy as np


import torch.autograd as grad


import torch.nn.functional as F


class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden,
            num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())
        x = x[:, (x.size(1) - 1)]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[(same_idx), :, (same_idx)]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-06).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.reshape(sum_centroids.shape[0], 1,
        sum_centroids.shape[-1])
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.view(utterance_centroids
        .shape[0] * utterance_centroids.shape[1], -1)
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[
        0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.
        shape[0], 1)
    embeddings_expand = embeddings_expand.view(embeddings_expand.shape[0] *
        embeddings_expand.shape[1], embeddings_expand.shape[-1])
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.
        size(0))
    same_idx = list(range(embeddings.size(0)))
    cos_diff[(same_idx), :, (same_idx)] = cos_same.view(embeddings.shape[0],
        num_utterances)
    cos_diff = cos_diff + 1e-06
    return cos_diff


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True
            )
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True
            )
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-06)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_HarryVolek_PyTorch_Speaker_Verification(_paritybench_base):
    pass
