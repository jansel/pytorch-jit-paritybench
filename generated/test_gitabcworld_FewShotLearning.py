import sys
_module = sys.modules[__name__]
del sys
config = _module
baselines = _module
imagenet = _module
lstm = _module
datasets = _module
miniImagenet = _module
logger = _module
main = _module
model = _module
bnlstm = _module
learner = _module
lstmhelper = _module
metaLearner = _module
metalstm = _module
recurrentLSTMNetwork = _module
option = _module
utils = _module
create_miniImagenet = _module
util = _module
visualize = _module

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


import numpy as np


from torch.autograd import Variable


import math


import torch.nn.init as init


from torch import nn


from torch.nn import functional


from torch.nn import init


class MatchingNet(nn.Module):

    def __init__(self, opt):
        super(MatchingNet, self).__init__()
        self.cosineSim = nn.CosineSimilarity()
        self.embedModel = importlib.import_module(opt['embedModel']).build(opt)
        self.embedModel.setCuda(opt['useCUDA'])
        self.lossF = nn.CrossEntropyLoss()

    def set(self, mode):
        self.embedModel.set(mode)

    def forward(self, opt, input):
        trainInput = input['trainInput']
        trainTarget = input['trainTarget']
        testInput = input['testInput']
        testTarget = input['testTarget']
        trainTarget = trainTarget.view(-1, 1)
        y_one_hot = trainTarget.clone()
        y_one_hot = y_one_hot.expand(trainTarget.size()[0], opt['nClasses']
            ['train'])
        y_one_hot.data.zero_()
        y_one_hot = y_one_hot.float().scatter_(1, trainTarget, 1)
        gS = self.embedModel.embedG(trainInput)
        fX = self.embedModel.embedF(testInput, gS, opt['steps'])
        repeatgS = gS.repeat(fX.size(0), 1)
        repeatfX = fX.repeat(1, gS.size(0)).view(fX.size(0) * gS.size(0),
            fX.size(1))
        weights = self.cosineSim(repeatgS, repeatfX).view(fX.size(0), gS.
            size(0), 1)
        expandOneHot = y_one_hot.view(1, y_one_hot.size(0), y_one_hot.size(1)
            ).expand(fX.size(0), y_one_hot.size(0), y_one_hot.size(1))
        expandWeights = weights.expand_as(expandOneHot)
        out = expandOneHot.mul(expandWeights).sum(1)
        if self.embedModel.isTraining():
            loss = self.lossF(out, testTarget)
            return out, loss
        else:
            return out


def convLayer(opt, nInput, nOutput, k):
    """3x3 convolution with padding"""
    seq = nn.Sequential(nn.Conv2d(nInput, nOutput, kernel_size=k, stride=1,
        padding=1, bias=True), nn.BatchNorm2d(nOutput), nn.ReLU(True), nn.
        MaxPool2d(kernel_size=2, stride=2))
    if opt['useDropout']:
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq


class Classifier(nn.Module):

    def __init__(self, opt):
        super(Classifier, self).__init__()
        finalSize = int(math.floor(opt['nIn'] / (2 * 2 * 2 * 2)))
        self.layer1 = convLayer(opt, 0, opt['nDepth'], opt['nFilters'], 3)
        self.layer2 = convLayer(opt, 1, opt['nFilters'], opt['nFilters'], 3)
        self.layer3 = convLayer(opt, 2, opt['nFilters'], opt['nFilters'], 3)
        self.layer4 = convLayer(opt, 3, opt['nFilters'], opt['nFilters'], 3)
        self.outSize = opt['nFilters'] * finalSize * finalSize
        self.classify = opt['classify']
        if self.classify:
            self.layer5 = nn.Linear(opt['nFilters'] * finalSize * finalSize,
                opt['nClasses']['train'])
        self.outSize = opt['nClasses']['train']
        self.reset()

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reset(self):
        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def forward(self, x):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.classify:
            x = self.layer5(x)
        return x


class SeparatedBatchNorm1d(nn.Module):
    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-05, momentum=0.1,
        affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """
        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer('running_mean_{}'.format(i), torch.zeros(
                num_features))
            self.register_buffer('running_var_{}'.format(i), torch.ones(
                num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'.format(
                input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(input=input_, running_mean=
            running_mean, running_var=running_var, weight=self.weight, bias
            =self.bias, training=self.training, momentum=self.momentum, eps
            =self.eps)

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, momentum={momentum}, max_length={max_length}, affine={affine})'
            .format(name=self.__class__.__name__, **self.__dict__))


class LSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 *
            hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 *
            hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.
            size())
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi, split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BNLSTMCell(nn.Module):
    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):
        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 *
            hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 *
            hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.bn_ih = SeparatedBatchNorm1d(num_features=4 * hidden_size,
            max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(num_features=4 * hidden_size,
            max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(num_features=hidden_size,
            max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        init.constant(self.bias.data, val=0)
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            time: The current timestep value, which is used to
                get appropriate running statistics.
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.
            size())
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch, split_size=
            self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        return h_1, c_1


class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
        use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size, hidden_size=
                hidden_size, **kwargs)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            if isinstance(cell, BNLSTMCell):
                h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            else:
                h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next * mask + hx[0] * (1 - mask)
            c_next = c_next * mask + hx[1] * (1 - mask)
            hx_next = h_next, c_next
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                length = length
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_()
                )
            hx = hx, hx
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(cell=
                self.cells[layer], input_=input_, length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)


class Learner(nn.Module):

    def __init__(self, opt):
        super(Learner, self).__init__()
        opt['nFilters'] = 32
        self.bn_layers = []
        for i in range(4):
            if 'BN_momentum' in opt.keys():
                self.bn_layers.append(nn.BatchNorm2d(opt['nFilters'],
                    momentum=opt['BN_momentum']))
            else:
                self.bn_layers.append(nn.BatchNorm2d(opt['nFilters']))
        opt['bnorm2d'] = self.bn_layers
        self.model = importlib.import_module(opt['learner']).build(opt)
        self.modelF = importlib.import_module(opt['learner']).build(opt)
        self.nParams = self.modelF.nParams
        self.params = {param[0]: param[1] for param in self.modelF.net.
            named_parameters()}

    def unflattenParams_net(self, flatParams):
        flatParams = flatParams.squeeze()
        indx = 0
        for param in self.model.net.parameters():
            lengthParam = param.view(-1).size()[0]
            param = flatParams[indx:lengthParam].view_as(param).clone()

    def forward(self, inputs, targets):
        output = self.modelF.net(inputs)
        loss = self.modelF.criterion(output, targets)
        return output, loss

    def feval(self, inputs, targets):
        self.model.net.zero_grad()
        outputs = self.model.net(inputs)
        loss = self.model.criterion(outputs, targets)
        loss.backward()
        grads = torch.cat([param.grad.view(-1) for param in self.model.net.
            parameters()], 0)
        return grads, loss

    def reset(self):
        self.model.net.reset()
        self.modelF.net.reset()

    def set(self, mode):
        if mode == 'training':
            self.model.net.train()
            self.modelF.net.train()
        elif mode == 'evaluate':
            self.model.net.eval()
            self.modelF.net.eval()
        else:
            None

    def setCuda(self, value=True):
        if value == True:
            self.model.net
            self.modelF.net
        else:
            self.model.net.cpu()
            self.modelF.net.cpu()


class MetaLearner(nn.Module):

    def __init__(self, opt):
        super(MetaLearner, self).__init__()
        self.nHidden = opt['nHidden'] if 'nHidden' in opt.keys() else 20
        self.maxGradNorm = opt['maxGradNorm'] if 'maxGradNorm' in opt.keys(
            ) else 0.25
        inputFeatures = 2
        batchNormalization1 = opt['BN1'] if 'BN1' in opt.keys() else False
        maxBatchNormalizationLayers = opt['steps'] if 'steps' in opt.keys(
            ) else 1
        batchNormalization1 = False
        if batchNormalization1:
            self.lstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell,
                input_size=inputFeatures, hidden_size=self.nHidden,
                batch_first=True, max_length=maxBatchNormalizationLayers)
        else:
            self.lstm = nn.LSTM(input_size=inputFeatures, hidden_size=self.
                nHidden, batch_first=True, num_layers=
                maxBatchNormalizationLayers)
        batch_size = 1
        self.lstm_h0_c0 = None
        batchNormalization2 = opt['BN2'] if 'BN2' in opt.keys() else False
        self.lstm2 = metalstm.MetaLSTM(input_size=opt['nParams'],
            hidden_size=self.nHidden, batch_first=True, num_layers=
            maxBatchNormalizationLayers)
        batch_size = 1
        self.lstm2_fS_iS_cS_deltaS = None
        self.params = lambda : list(self.lstm.named_parameters()) + list(self
            .lstm2.named_parameters())
        self.params = {param[0]: param[1] for param in self.params()}
        for names in self.lstm._all_weights:
            for name in filter(lambda n: 'weight' in n, names):
                weight = getattr(self.lstm, name)
                weight.data.uniform_(-0.01, 0.01)
        for params in self.lstm2.named_parameters():
            if 'WF' in names[0] or names[0] in names[0] or 'cI' in params[0]:
                params[1].data.uniform_(-0.01, 0.01)
        for params in self.lstm2.named_parameters():
            if 'cell_0.bF' in names[0]:
                params[0].data.uniform_(4, 5)
            if 'cell_0.bI' in names[0]:
                params[0].data.uniform_(-4, -5)
        initialParams = torch.cat([value.view(-1) for key, value in opt[
            'learnerParams'].items()], 0)
        for params in self.lstm2.named_parameters():
            if 'cell_0.cI' in params[0]:
                params[1].data = initialParams.data.clone()
        a = 0

    def forward(self, learner, trainInput, trainTarget, testInput,
        testTarget, steps, batchSize, evaluate=False):
        trainSize = trainInput.size(0)
        learner.reset()
        learner.set('training')
        util.unflattenParams(learner.model, self.lstm2.cells[0].cI)
        idx = 0
        for s in range(steps):
            for i in range(0, trainSize, batchSize):
                x = trainInput[i:batchSize, :]
                y = trainTarget[i:batchSize]
                grad_model, loss_model = learner.feval(x, y)
                grad_model = grad_model.view(grad_model.size()[0], 1, 1)
                inputs = torch.cat((grad_model, loss_model.expand_as(
                    grad_model)), 2)
                """
                # preprocess grad & loss by DeepMind "Learning to learn"
                preGrad, preLoss = preprocess(grad_model,loss_model)
                # use meta-learner to get learner's next parameters
                lossExpand = preLoss.expand_as(preGrad)
                inputs = torch.cat((lossExpand,preGrad),2)
                """
                output, self.lstm_h0_c0 = self.lstm(inputs, self.lstm_h0_c0)
                self.lstm2_fS_iS_cS_deltaS = self.lstm2((output, grad_model
                    ), self.lstm2_fS_iS_cS_deltaS)
                util.unflattenParams(learner.modelF, self.
                    lstm2_fS_iS_cS_deltaS[2])
                output, loss = learner(testInput, testTarget)
                output = self.lstm2_fS_iS_cS_deltaS[2]
                util.unflattenParams(learner.model, output)
                idx = idx + 1
        util.unflattenParams(learner.modelF, output)
        if evaluate:
            learner.set('evaluate')
        output, loss = learner(testInput, testTarget)
        torch.autograd.grad(loss, self.lstm2.parameters())
        return output, loss

    def gradNorm(self, loss):
        None
        for params in self.lstm.parameters():
            None
        for params in self.lstm2.parameters():
            None
        a = 0

    def setCuda(self, value=True):
        if value:
            self.lstm
            self.lstm2
        else:
            self.lstm.cpu()
            self.lstm2.cpu()


class MetaLSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(MetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.WF = nn.Parameter(torch.FloatTensor(hidden_size + 2, 1))
        self.WI = nn.Parameter(torch.FloatTensor(hidden_size + 2, 1))
        self.cI = nn.Parameter(torch.FloatTensor(input_size, 1))
        self.bI = nn.Parameter(torch.FloatTensor(1, 1))
        self.bF = nn.Parameter(torch.FloatTensor(1, 1))
        self.m = nn.Parameter(torch.FloatTensor(1))
        """
        self.WF = Variable(torch.FloatTensor(hidden_size + 2, 1), requires_grad=True)
        self.WI = Variable(torch.FloatTensor(hidden_size + 2, 1), requires_grad=True)
        # initial cell state is a param
        self.cI = Variable(torch.FloatTensor(input_size, 1), requires_grad=True)
        self.bI = Variable(torch.FloatTensor(1, 1), requires_grad=True)
        self.bF = Variable(torch.FloatTensor(1, 1), requires_grad=True)
        self.m = Variable(torch.FloatTensor(1), requires_grad=True)
        """
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters 
        """
        self.WF.data.uniform_(-0.01, 0.01)
        self.WI.data.uniform_(-0.01, 0.01)
        self.cI.data.uniform_(-0.01, 0.01)
        self.bI.data.zero_()
        self.bF.data.zero_()
        self.m.data.zero_()

    def forward(self, input_, grads_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        fS, iS, cS, deltaS = hx
        fS = torch.cat((cS, fS), 1)
        iS = torch.cat((cS, iS), 1)
        fS = torch.mm(torch.cat((input_, fS), 1), self.WF)
        fS += self.bF.expand_as(fS)
        iS = torch.mm(torch.cat((input_, iS), 1), self.WI)
        iS += self.bI.expand_as(iS)
        deltaS = self.m * deltaS - nn.Sigmoid()(iS).mul(grads_)
        cS = nn.Sigmoid()(fS).mul(cS) + deltaS
        return fS, iS, cS, deltaS

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MetaLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1
        ):
        super(MetaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = MetaLSTMCell(input_size=layer_input_size, hidden_size=
                hidden_size)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, grads_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            hx = cell(input_=input_[time], grads_=grads_[time], hx=hx)
        return hx

    def forward(self, input_, length=None, hx=None):
        x_input = input_[0]
        grad_input = input_[1]
        if self.batch_first:
            x_input = x_input.transpose(0, 1)
            grad_input = grad_input.transpose(0, 1)
        max_time, batch_size, _ = x_input.data.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if x_input.is_cuda:
                length = length
        if hx is None:
            fS = Variable(grad_input.data.new(batch_size, 1).zero_())
            iS = Variable(grad_input.data.new(batch_size, 1).zero_())
            cS = self.cells[0].cI.unsqueeze(1)
            deltaS = Variable(grad_input.data.new(batch_size, 1).zero_())
            hx = fS, iS, cS, deltaS
        fS_n = []
        iS_n = []
        cS_n = []
        deltaS_n = []
        for layer in range(self.num_layers):
            hx_new = MetaLSTM._forward_rnn(cell=self.cells[layer], input_=
                x_input, grads_=grad_input, length=length, hx=hx)
            fS_n.append(hx_new[0])
            iS_n.append(hx_new[1])
            cS_n.append(hx_new[2])
            deltaS_n.append(hx_new[3])
        fS_n = torch.stack(fS_n, 0)
        iS_n = torch.stack(iS_n, 0)
        cS_n = torch.stack(cS_n, 0)
        fS_n = torch.stack(fS_n, 0)
        deltaS_n = torch.stack(deltaS_n, 0)
        return fS_n, iS_n, cS_n, deltaS_n


class RecurrentLSTMNetwork(nn.Module):

    def __init__(self, opt):
        super(RecurrentLSTMNetwork, self).__init__()
        self.inputFeatures = opt['inputFeatures'
            ] if 'inputFeatures' in opt.keys() else 10
        self.hiddenFeatures = opt['hiddenFeatures'
            ] if 'hiddenFeatures' in opt.keys() else 100
        self.outputType = opt['outputType'] if 'outputType' in opt.keys(
            ) else 'last'
        self.batchNormalization = opt['batchNormalization'
            ] if 'batchNormalization' in opt.keys() else False
        self.maxBatchNormalizationLayers = opt['maxBatchNormalizationLayers'
            ] if 'batchNormalization' in opt.keys() else 10
        self.p = {}
        self.p['W'] = Variable(torch.zeros(self.inputFeatures + self.
            hiddenFeatures, 4 * self.hiddenFeatures), requires_grad=True)
        self.params = [self.p['W']]
        self.batchNormalization = True
        if self.batchNormalization:
            lstm_bn = nn.BatchNorm1d(4 * self.hiddenFeatures)
            cell_bn = nn.BatchNorm1d(self.hiddenFeatures)
            self.layers = {'lstm_bn': [lstm_bn], 'cell_bn': [cell_bn]}
            for i in range(2, self.maxBatchNormalizationLayers):
                lstm_bn = nn.BatchNorm1d(4 * self.hiddenFeatures)
                cell_bn = nn.BatchNorm1d(self.hiddenFeatures)
                self.layers['lstm_bn'].append(lstm_bn)
                self.layers['cell_bn'].append(cell_bn)
            self.layers['lstm_bn'][0].weight.data.fill_(0.1)
            self.layers['lstm_bn'][0].bias.data.zero_()
            self.layers['cell_bn'][0].weight.data.fill_(0.1)
            self.layers['cell_bn'][0].bias.data.zero_()
            self.params = self.params + list(self.layers['lstm_bn'][0].
                parameters()) + list(self.layers['lstm_bn'][0].parameters())
        else:
            self.p['b'] = Variable(torch.zeros(1, 4 * self.hiddenFeatures),
                require_grad=True)
            self.params = self.params + [self.p['b']]
            self.layers = {}

    def setCuda(self, value=True):
        if value == True:
            for key in self.p.keys():
                self.p[key]
            for key in self.layers.keys():
                for i in range(len(self.layers[key])):
                    self.layers[key][i]
        else:
            for key in self.p.keys():
                self.p[key].cpu()
            for key in self.layers.keys():
                for i in range(len(self.layers[key])):
                    self.layers[key][i].cpu()

    def forward(self, x, prevState=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        batch = x.size(0)
        steps = x.size(1)
        if prevState == None:
            prevState = {}
        hs = {}
        cs = {}
        for t in range(steps):
            xt = x[:, (t), :]
            hp = hs[t - 1] or prevState.h or torch.zeros()
        a = 0


class Classifier(nn.Module):

    def __init__(self, opt):
        super(Classifier, self).__init__()
        nFilters = 64
        finalSize = int(math.floor(opt['nIn'] / (2 * 2 * 2 * 2)))
        self.layer1 = convLayer(opt, opt['nDepth'], nFilters, 3)
        self.layer2 = convLayer(opt, nFilters, nFilters, 3)
        self.layer3 = convLayer(opt, nFilters, nFilters, 3)
        self.layer4 = convLayer(opt, nFilters, nFilters, 3)
        self.outSize = nFilters * finalSize * finalSize
        self.classify = opt['classify']
        if self.classify:
            self.layer5 = nn.Linear(nFilters * finalSize * finalSize, opt[
                'nClasses']['train'])
            self.outSize = opt['nClasses']['train']
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

    def forward(self, x):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.classify:
            x = self.layer5(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gitabcworld_FewShotLearning(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(SeparatedBatchNorm1d(*[], **{'num_features': 4, 'max_length': 4}), [torch.rand([4, 4, 4, 4]), 0], {})
