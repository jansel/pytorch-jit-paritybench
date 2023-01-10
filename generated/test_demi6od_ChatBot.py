import sys
_module = sys.modules[__name__]
del sys
model = _module
run = _module
train = _module
model = _module
run = _module
train = _module
ChatBot = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import random


import torch.optim as optim


import math


import time


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransBertEncoder(nn.Module):

    def __init__(self, nhead=8, nlayers=6, dropout=0.5):
        super().__init__()
        self.bert = g_bert
        self.pos_encoder = PositionalEncoding(g_bert_emb_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=g_bert_emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        with torch.no_grad():
            embedded = self.bert(src.transpose(0, 1))[0].transpose(0, 1)
        outputs = self.transformer_encoder(embedded)
        return outputs


g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransGptDecoder(nn.Module):

    def __init__(self, nhead=8, nlayers=6):
        super().__init__()
        self.gpt = g_gpt
        decoder_layer = nn.TransformerDecoderLayer(d_model=g_gpt_emb_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

    def forward(self, meaning, tgt, output_len, teacher_forcing_ratio):
        batch_size = meaning.size(1)
        context = self.transformer_decoder(meaning, memory=meaning)
        teacher_force = random.random() < teacher_forcing_ratio
        past = None
        predictions = torch.zeros(output_len, batch_size, g_gpt_vocab_size)
        for t in range(output_len):
            if t == 0:
                output, past = self.gpt(input_ids=None, inputs_embeds=context.transpose(0, 1), past=past)
            else:
                if teacher_force and self.training:
                    context = tgt[t].unsqueeze(0)
                output, past = self.gpt(context.transpose(0, 1), past=past)
            output = output.transpose(0, 1)
            predictions[t] = output[-1]
            token = torch.argmax(output[-1], 1)
            context = token.unsqueeze(0)
        return predictions


class GruEncoder(nn.Module):
    """compress the request embeddings to meaning"""

    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        return hidden


class GruDecoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt, hidden):
        fc_output = src[0].unsqueeze(0)
        tgt_len = tgt.size(0)
        batch_size = tgt.size(1)
        outputs = torch.zeros(tgt_len, batch_size, g_bert_emb_dim)
        for t in range(0, tgt_len):
            gru_output, hidden = self.gru(fc_output, hidden)
            fc_output = self.fc(gru_output)
            outputs[t] = fc_output
        return outputs


class DialogDNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = self.dropout(input)
        output = input + F.relu(self.fc1(input))
        output = self.dropout(output)
        output = output + F.relu(self.fc2(output))
        output = self.dropout(output)
        output = output + self.fc3(output)
        return output


PRINT_CHAT = True


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def print_chat(sentences):
    None
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        max_idx_t = word_embed.argmax()
        max_idx = max_idx_t.item()
        word = TRG.vocab.itos[max_idx]
        None
    None


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, 'Hidden dimensions of encoder and decoder must be equal!'

    def forward(self, src, trg, teacher_forcing_ratio=0.8):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        context = self.encoder(src)
        hidden = context
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        if PRINT_CHAT:
            print_chat(outputs)
        return outputs


class TransBertDecoder(nn.Module):

    def __init__(self, nhead=8, nlayers=6, dropout=0.5):
        super().__init__()
        self.bert = g_bert
        self.pos_decoder = PositionalEncoding(g_bert_emb_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=g_bert_emb_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.fc_out = nn.Linear(g_bert_emb_dim, g_vocab_size)

    def forward(self, tgt, meaning, teacher_forcing_ratio):
        output_len = tgt.size(0)
        batch_size = tgt.size(1)
        teacher_force = random.random() < teacher_forcing_ratio
        if teacher_force and self.training:
            tgt_emb_total = torch.zeros(output_len, batch_size, g_bert_emb_dim)
            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(tgt[:t + 1].transpose(0, 1))[0].transpose(0, 1)
                tgt_emb_total[t] = tgt_emb[-1]
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb_total))
            decoder_output = self.transformer_decoder(tgt=tgt_emb_total, memory=meaning, tgt_mask=tgt_mask)
            predictions = self.fc_out(decoder_output)
        else:
            output = torch.full((output_len + 1, batch_size), g_tokenizer.cls_token_id, dtype=torch.long, device=g_device)
            predictions = torch.zeros(output_len, batch_size, g_vocab_size)
            for t in range(0, output_len):
                with torch.no_grad():
                    tgt_emb = self.bert(output[:t + 1].transpose(0, 1))[0].transpose(0, 1)
                tgt_mask = nn.Transformer().generate_square_subsequent_mask(len(tgt_emb))
                decoder_output = self.transformer_decoder(tgt=tgt_emb, memory=meaning, tgt_mask=tgt_mask)
                prediction = self.fc_out(decoder_output[-1])
                predictions[t] = prediction
                one_hot_idx = prediction.argmax(1)
                output[t + 1] = one_hot_idx
        return predictions


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        weight_matrix = SRC.vocab.vectors
        self.embedding.weight.data.copy_(weight_matrix)
        self.rnn = nn.GRU(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        weight_matrix = TRG.vocab.vectors
        self.embedding.weight.data.copy_(weight_matrix)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        output = self.dropout(output)
        prediction = self.fc_out(output)
        return prediction, hidden


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DialogDNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GruEncoder,
     lambda: ([], {'hidden_size': 4, 'input_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_demi6od_ChatBot(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

