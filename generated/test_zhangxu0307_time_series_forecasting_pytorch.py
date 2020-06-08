import sys
_module = sys.modules[__name__]
del sys
FCD_model = _module
decomposition_net = _module
plot_diff = _module
plot_loss = _module
HMM = _module
main = _module
svm_forecasting = _module
ARIMA = _module
Holt_Winters = _module
ML_forecasting = _module
NN_forecasting = _module
NN_train = _module
eval = _module
model = _module
ts_decompose = _module
ts_loader = _module
util = _module

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


from torch.autograd import Variable


import torch


import torch.autograd as autograd


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell,
        use_cuda=False):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.use_cuda = use_cuda
        if cell == 'RNN':
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.
                hiddenNum, num_layers=self.layerNum, dropout=0.0,
                nonlinearity='tanh', batch_first=True)
        if cell == 'LSTM':
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.
                hiddenNum, num_layers=self.layerNum, dropout=0.0,
                batch_first=True)
        if cell == 'GRU':
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.
                hiddenNum, num_layers=self.layerNum, dropout=0.0,
                batch_first=True)
        None
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


class ResRNN_Cell(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, use_cuda=False
        ):
        super(ResRNN_Cell, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.use_cuda = use_cuda
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)
            )
        if self.use_cuda:
            h0 = h0
        ht = h0
        lag = x.data.size()[1]
        outputs = []
        for i in range(lag):
            hn = self.i2h(x[:, (i), :]) + self.h2h(h0)
            if i == 0:
                hstart = hn
            elif i == lag - 2:
                h0 = nn.Tanh()(hn + hstart)
            elif self.resDepth == 1:
                h0 = nn.Tanh()(hn + h0)
            elif i % self.resDepth == 0:
                h0 = nn.Tanh()(hn + ht)
                ht = hn
            else:
                h0 = nn.Tanh()(hn)
            outputs.append(hn)
        output_hiddens = torch.cat(outputs, 0)
        return output_hiddens


class ResRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, use_cuda=False
        ):
        super(ResRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.use_cuda = use_cuda
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.i2h = self.i2h
        self.h2h = self.h2h
        self.h2o = self.h2o
        self.fc = self.fc
        self.ht2h = self.ht2h

    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)
            )
        if self.use_cuda:
            h0 = h0
        lag = x.data.size()[1]
        ht = h0
        for i in range(lag):
            hn = self.i2h(x[:, (i), :]) + self.h2h(h0)
            if i == 0:
                hstart = hn
            elif i == lag - 1:
                h0 = nn.Tanh()(hn + hstart)
            elif self.resDepth == 1:
                h0 = nn.Tanh()(hn + h0)
            elif i % self.resDepth == 0:
                h0 = nn.Tanh()(hn + ht)
                ht = hn
            else:
                h0 = nn.Tanh()(hn)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)
        return fcOutput


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class RNN_Attention(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, seq_len,
        merge='concate', use_cuda=True):
        super(RNN_Attention, self).__init__()
        self.att_fc = nn.Linear(hiddenNum, 1)
        self.time_distribut_layer = TimeDistributed(self.att_fc)
        if merge == 'mean':
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == 'concate':
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        self.cell = ResRNN_Cell(inputDim, hiddenNum, outputDim, resDepth,
            use_cuda=use_cuda)
        if use_cuda:
            self.cell = self.cell

    def forward(self, x):
        batchSize = x.size(0)
        rnnOutput = self.cell(x)
        attention_out = self.time_distribut_layer(rnnOutput)
        attention_out = attention_out.view((batchSize, -1))
        attention_out = F.softmax(attention_out)
        attention_out = attention_out.view(-1, batchSize, 1)
        rnnOutput = rnnOutput * attention_out
        if self.merge == 'mean':
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == 'concate':
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
        fcOutput = self.dense(x)
        return fcOutput


class MLPModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim):
        super(MLPModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.fc1 = nn.Linear(self.inputDim, self.hiddenNum)
        self.fc2 = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhangxu0307_time_series_forecasting_pytorch(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(ResRNN_Cell(*[], **{'inputDim': 4, 'hiddenNum': 4, 'outputDim': 4, 'resDepth': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(RNN_Attention(*[], **{'inputDim': 4, 'hiddenNum': 4, 'outputDim': 4, 'resDepth': 4, 'seq_len': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(MLPModel(*[], **{'inputDim': 4, 'hiddenNum': 4, 'outputDim': 4}), [torch.rand([4, 4, 4, 4])], {})
