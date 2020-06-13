import sys
_module = sys.modules[__name__]
del sys
Data = _module
GST = _module
Hyperparameters = _module
Loss = _module
Modules = _module
Network = _module
Synthesis = _module
blizzard_preprocess = _module
cutoff = _module
generate = _module
preprocess = _module
train = _module
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


import torch.nn.init as init


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.nn import DataParallel


from scipy.io.wavfile import write


class GST(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL()

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)
        return style_embed


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self):
        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i], out_channels=filters[i +
            1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) for i in
            range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.
            ref_enc_filters[i]) for i in range(K)])
        out_channels = self.calculate_channels(hp.n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
            hidden_size=hp.E // 2, batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.n_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
        out = out.transpose(1, 2)
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)
        self.gru.flatten_parameters()
        memory, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    """
    inputs --- [N, E//2]
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.E //
            hp.num_heads))
        d_q = hp.E // 2
        d_k = hp.E // hp.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k,
            num_units=hp.E, num_heads=hp.num_heads)
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        style_embed = self.attention(query, keys)
        return style_embed


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=
            num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units,
            bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=
            num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / self.key_dim ** 0.5
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        return out


class MultiHeadAttention2(nn.Module):

    def __init__(self, query_dim, key_dim, num_units, h=8, is_masked=False):
        super(MultiHeadAttention, self).__init__()
        self._num_units = num_units
        self._h = h
        self._key_dim = key_dim
        self._is_masked = is_masked
        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)
        attention = torch.matmul(Q, K.transpose(1, 2))
        attention = attention / self._key_dim ** 0.5
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2 ** 32 + 1)
            attention = attention * diag_mat + mask * (diag_mat - 1).abs()
        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, V)
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(attention.split(split_size=restore_chunk_size,
            dim=0), dim=2)
        return attention


class TacotronLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mels, mels_hat, mags, mags_hat):
        mel_loss = torch.mean(torch.abs(mels - mels_hat))
        mag_loss = torch.abs(mags - mags_hat)
        mag_loss = 0.5 * torch.mean(mag_loss) + 0.5 * torch.mean(mag_loss[:,
            :, :hp.n_priority_freq])
        return mel_loss, mag_loss


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='same'):
        """
        inputs: [N, T, C_in]
        outputs: [N, T, C_out]
        """
        super().__init__()
        if padding == 'same':
            left = (kernel_size - 1) // 2
            right = kernel_size - 1 - left
            self.pad = left, right
        else:
            self.pad = 0, 0
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        inputs = F.pad(inputs, self.pad)
        out = self.conv1d(inputs)
        out = torch.transpose(out, 1, 2)
        return out


class Highway(nn.Module):

    def __init__(self, in_features, out_features):
        """
        inputs: [N, T, C]
        outputs: [N, T, C]
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        H = self.linear1(inputs)
        H = F.relu(H)
        T = self.linear2(inputs)
        T = F.sigmoid(T)
        out = H * T + inputs * (1.0 - T)
        return out


class Conv1dBank(nn.Module):
    """
        inputs: [N, T, C_in]
        outputs: [N, T, C_out * K]  # same padding
    Args:
        in_channels: E//2
        out_channels: E//2
    """

    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.bank = nn.ModuleList()
        for k in range(1, K + 1):
            self.bank.append(Conv1d(in_channels, out_channels, kernel_size=k))
        self.bn = BatchNorm1d(out_channels * K)

    def forward(self, inputs):
        outputs = self.bank[0](inputs)
        for k in range(1, len(self.bank)):
            output = self.bank[k](inputs)
            outputs = torch.cat([outputs, output], dim=2)
        outputs = self.bn(outputs)
        outputs = F.relu(outputs)
        return outputs


class BatchNorm1d(nn.Module):
    """
    inputs: [N, T, C]
    outputs: [N, T, C]
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        out = self.bn(inputs.transpose(1, 2).contiguous())
        return out.transpose(1, 2)


class PreNet(nn.Module):
    """
    inputs: [N, T, in]
    outputs: [N, T, E // 2]
    """

    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hp.E)
        self.linear2 = nn.Linear(hp.E, hp.E // 2)
        self.dropout1 = nn.Dropout(hp.dropout_p)
        self.dropout2 = nn.Dropout(hp.dropout_p)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = F.relu(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout2(outputs)
        return outputs


class AttentionRNN(nn.Module):
    """
    input:
        inputs: [N, T_y, E//2]
        memory: [N, T_x, E]

    output:
        attn_weights: [N, T_y, T_x]
        outputs: [N, T_y, E]
        hidden: [1, N, E]

    T_x --- character len
    T_y --- spectrogram len
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E,
            batch_first=True, bidirectional=False)
        self.W = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.U = nn.Linear(in_features=hp.E, out_features=hp.E, bias=False)
        self.v = nn.Linear(in_features=hp.E, out_features=1, bias=False)

    def forward(self, inputs, memory, prev_hidden=None):
        T_x = memory.size(1)
        T_y = inputs.size(1)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(inputs, prev_hidden)
        w = self.W(outputs).unsqueeze(2).expand(-1, -1, T_x, -1)
        u = self.U(memory).unsqueeze(1).expand(-1, T_y, -1, -1)
        attn_weights = self.v(F.tanh(w + u).view(-1, hp.E)).view(-1, T_y, T_x)
        attn_weights = F.softmax(attn_weights, 2)
        return attn_weights, outputs, hidden


class Tacotron(nn.Module):
    """
    input:
        texts: [N, T_x]
        mels: [N, T_y/r, n_mels*r]
    output:
        mels --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(hp.vocab), hp.E)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.gst = GST()

    def forward(self, texts, mels, ref_mels):
        embedded = self.embedding(texts)
        memory, encoder_hidden = self.encoder(embedded)
        style_embed = self.gst(ref_mels)
        style_embed = style_embed.expand_as(memory)
        memory = memory + style_embed
        mels_hat, mags_hat, attn_weights = self.decoder(mels, memory)
        return mels_hat, mags_hat, attn_weights


def max_pool1d(inputs, kernel_size, stride=1, padding='same'):
    """
    inputs: [N, T, C]
    outputs: [N, T // stride, C]
    """
    inputs = inputs.transpose(1, 2)
    if padding == 'same':
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        pad = left, right
    else:
        pad = 0, 0
    inputs = F.pad(inputs, pad)
    outputs = F.max_pool1d(inputs, kernel_size, stride)
    outputs = outputs.transpose(1, 2)
    return outputs


class Encoder(nn.Module):
    """
    input:
        inputs: [N, T_x, E]
    output:
        outputs: [N, T_x, E]
        hidden: [2, N, E//2]
    """

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(in_features=hp.E)
        self.conv1d_bank = Conv1dBank(K=hp.K, in_channels=hp.E // 2,
            out_channels=hp.E // 2)
        self.conv1d_1 = Conv1d(in_channels=hp.K * hp.E // 2, out_channels=
            hp.E // 2, kernel_size=3)
        self.conv1d_2 = Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 
            2, kernel_size=3)
        self.bn1 = BatchNorm1d(num_features=hp.E // 2)
        self.bn2 = BatchNorm1d(num_features=hp.E // 2)
        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.E // 2,
                out_features=hp.E // 2))
        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2,
            num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        inputs = self.prenet(inputs)
        outputs = self.conv1d_bank(inputs)
        outputs = max_pool1d(outputs, kernel_size=2)
        outputs = self.conv1d_1(outputs)
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.conv1d_2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + inputs
        for layer in self.highways:
            outputs = layer(outputs)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)
        return outputs, hidden


class Decoder(nn.Module):
    """
    input:
        inputs --- [N, T_y/r, n_mels * r]
        memory --- [N, T_x, E]
    output:
        mels   --- [N, T_y/r, n_mels*r]
        mags --- [N, T_y, 1+n_fft//2]
        attn_weights --- [N, T_y/r, T_x]
    """

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(hp.n_mels)
        self.attn_rnn = AttentionRNN()
        self.attn_projection = nn.Linear(in_features=2 * hp.E, out_features
            =hp.E)
        self.gru1 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=
            True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=hp.E, hidden_size=hp.E, batch_first=
            True, bidirectional=False)
        self.fc1 = nn.Linear(in_features=hp.E, out_features=hp.n_mels * hp.r)
        self.cbhg = DecoderCBHG()
        self.fc2 = nn.Linear(in_features=hp.E, out_features=1 + hp.n_fft // 2)

    def forward(self, inputs, memory):
        if self.training:
            outputs = self.prenet(inputs)
            attn_weights, outputs, attn_hidden = self.attn_rnn(outputs, memory)
            attn_apply = torch.bmm(attn_weights, memory)
            attn_project = self.attn_projection(torch.cat([attn_apply,
                outputs], dim=2))
            self.gru1.flatten_parameters()
            outputs1, gru1_hidden = self.gru1(attn_project)
            gru_outputs1 = outputs1 + attn_project
            self.gru2.flatten_parameters()
            outputs2, gru2_hidden = self.gru2(gru_outputs1)
            gru_outputs2 = outputs2 + gru_outputs1
            mels = self.fc1(gru_outputs2)
            out, cbhg_hidden = self.cbhg(mels)
            mags = self.fc2(out)
            return mels, mags, attn_weights
        else:
            attn_hidden = None
            gru1_hidden = None
            gru2_hidden = None
            mels = []
            mags = []
            attn_weights = []
            for i in range(hp.max_Ty):
                inputs = self.prenet(inputs)
                attn_weight, outputs, attn_hidden = self.attn_rnn(inputs,
                    memory, attn_hidden)
                attn_weights.append(attn_weight)
                attn_apply = torch.bmm(attn_weight, memory)
                attn_project = self.attn_projection(torch.cat([attn_apply,
                    outputs], dim=-1))
                self.gru1.flatten_parameters()
                outputs1, gru1_hidden = self.gru1(attn_project, gru1_hidden)
                outputs1 = outputs1 + attn_project
                self.gru2.flatten_parameters()
                outputs2, gru2_hidden = self.gru2(outputs1, gru2_hidden)
                outputs2 = outputs2 + outputs1
                mel = self.fc1(outputs2)
                inputs = mel[:, :, -hp.n_mels:]
                mels.append(mel)
            mels = torch.cat(mels, dim=1)
            attn_weights = torch.cat(attn_weights, dim=1)
            out, cbhg_hidden = self.cbhg(mels)
            mags = self.fc2(out)
            return mels, mags, attn_weights


class DecoderCBHG(nn.Module):
    """
    input:
        inputs: [N, T/r, n_mels * r]
    output:
        outputs: [N, T, E]
        hidden: [2, N, E//2]
    """

    def __init__(self):
        super().__init__()
        self.conv1d_bank = Conv1dBank(K=hp.decoder_K, in_channels=hp.n_mels,
            out_channels=hp.E // 2)
        self.conv1d_1 = Conv1d(in_channels=hp.decoder_K * hp.E // 2,
            out_channels=hp.E, kernel_size=3)
        self.bn1 = BatchNorm1d(hp.E)
        self.conv1d_2 = Conv1d(in_channels=hp.E, out_channels=hp.n_mels,
            kernel_size=3)
        self.bn2 = BatchNorm1d(hp.n_mels)
        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.n_mels,
                out_features=hp.n_mels))
        self.gru = nn.GRU(input_size=hp.n_mels, hidden_size=hp.E // 2,
            num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        inputs = inputs.view(inputs.size(0), -1, hp.n_mels)
        outputs = self.conv1d_bank(inputs)
        outputs = max_pool1d(outputs, kernel_size=2)
        outputs = self.conv1d_1(outputs)
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)
        outputs = self.conv1d_2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + inputs
        for layer in self.highways:
            outputs = layer(outputs)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)
        return outputs, hidden


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KinglittleQ_GST_Tacotron(_paritybench_base):
    pass
    def test_000(self):
        self._check(BatchNorm1d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(Conv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Conv1dBank(*[], **{'K': 4, 'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(Highway(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MultiHeadAttention(*[], **{'query_dim': 4, 'key_dim': 4, 'num_units': 4, 'num_heads': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

