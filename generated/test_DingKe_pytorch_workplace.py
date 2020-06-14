import sys
_module = sys.modules[__name__]
del sys
cnn = _module
mlp = _module
optim = _module
custom_function = _module
custom_module = _module
functions = _module
modules = _module
reference = _module
adam = _module
cnn = _module
functions = _module
mlp = _module
modules = _module
build = _module
relu = _module
relu = _module
test = _module
dgc = _module
mlp = _module
mlp = _module
loss = _module
mnist_mlp = _module
cnn = _module
lstm = _module
mlp = _module
modules = _module
mlp = _module
modules = _module
test_gru = _module
test_indrnn = _module
test_lstm = _module
test_lstmon = _module
test_lstmp = _module
test_mgru = _module
test_rnn = _module
cnn = _module
modules = _module
activation = _module
mnist_mlp = _module

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


from torch.autograd import Variable


import math


from torch.autograd import Function


import torch.nn.functional as F


from torch.nn.modules.module import Module


from torch.nn import Module


from torch.nn import Parameter


eps = 1e-06


momentum = 0.9


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding
            =2), nn.BatchNorm2d(16, momentum=momentum, eps=eps), nn.
            MaxPool2d(2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5,
            padding=2), nn.BatchNorm2d(32, momentum=momentum, eps=eps), nn.
            MaxPool2d(2), nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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


class ReLUF(Function):

    @staticmethod
    def forward(cxt, input):
        cxt.save_for_backward(input)
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(cxt, grad_output):
        input, = cxt.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


relu = ReLUF.apply


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = relu(out)
        out = self.fc2(out)
        return out


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ReLU(nn.Module):

    def forward(self, input):
        return relu(input)


class LinearF(Function):

    @staticmethod
    def forward(cxt, input, weight, bias=None):
        cxt.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_variables
        grad_input = grad_weight = grad_bias = None
        if cxt.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if cxt.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


linear = LinearF.apply


class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True

    Shape:
        - Input: :math:`(N, in\\_features)`
        - Output: :math:`(N, out\\_features)`

    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Glorot Initialization
        """
        stdv = math.sqrt(2.0 / sum(self.weight.size()))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


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


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(BinaryConv2d(1, 16, kernel_size=5,
            padding=2), nn.BatchNorm2d(16, momentum=momentum, eps=eps), nn.
            MaxPool2d(2), BinaryTanh())
        self.layer2 = nn.Sequential(BinaryConv2d(16, 32, kernel_size=5,
            padding=2), nn.BatchNorm2d(32, momentum=momentum, eps=eps), nn.
            MaxPool2d(2), BinaryTanh())
        self.fc = BinaryLinear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


drop_hid = 0.0


drop_in = 0.0


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.p_in = nn.Dropout(p=drop_in)
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size
            layer = nn.Sequential(BinaryLinear(in_features, out_features,
                bias=False), nn.BatchNorm1d(out_features, momentum=momentum,
                eps=eps), BinaryTanh(), nn.Dropout(p=drop_hid))
            setattr(self, 'layer{}'.format(i), layer)
        self.fc = BinaryLinear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.p_in(x)
        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out


class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


binarize = BinarizeF.apply


class BinaryTanh(nn.Module):

    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output


class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1.0 / stdv


class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride, self.padding,
            self.dilation, self.groups)

    def reset_parameters(self):
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1.0 / stdv


class ReLUM(Module):

    def forward(self, input):
        return ReLUF()(input)


class MyNetwork(nn.Module):

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.relu = ReLUM()

    def forward(self, input):
        return self.relu(input)


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


class DNI(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DNI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.fc2(out)
        return out

    def reset_parameters(self):
        super(DNI, self).reset_parameters()
        for param in self.fc2.parameters():
            param.data.zero_()


class Net1(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Net1, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size), nn.
            BatchNorm1d(hidden_size), nn.ReLU())

    def forward(self, x):
        return self.mlp.forward(x)


class Net2(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net2, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(input_size, hidden_size))
        self.mlp.add_module('bn1', nn.BatchNorm1d(hidden_size))
        self.mlp.add_module('act1', nn.ReLU())
        self.mlp.add_module('fc', nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        return self.mlp.forward(x)


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.0
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum()


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


k = 100


simplified = False


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(meConv2d(1, 16, kernel_size=5, padding=
            2, k=k, simplified=simplified), nn.BatchNorm2d(16, momentum=
            momentum, eps=eps), nn.MaxPool2d(2), nn.ReLU())
        self.layer2 = nn.Sequential(meConv2d(16, 32, kernel_size=5, padding
            =2, k=k, simplified=simplified), nn.BatchNorm2d(32, momentum=
            momentum, eps=eps), nn.MaxPool2d(2), nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None, k=1, simplified=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = meLSTM(input_size, hidden_size, num_layers=num_layers,
            bias=bias, return_sequences=False, grad_clip=None, k=k,
            simplified=simplified)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        zeros = Variable(torch.zeros(x.size(0), self.hidden_size))
        initial_states = [(zeros, zeros)] * self.num_layers
        out, _ = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.p_in = nn.Dropout(p=drop_in)
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size
            layer = nn.Sequential(meLinear(in_features, out_features, bias=
                False, k=k, simplified=simplified), nn.BatchNorm1d(
                out_features, momentum=momentum, eps=eps), nn.ReLU(), nn.
                Dropout(p=drop_hid))
            setattr(self, 'layer{}'.format(i), layer)
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.p_in(x)
        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out


def simplified_topk(x, k):
    """ Proof-of-concept implementation of simplified topk
    Note all we neend the k-th largest vaule, thus an algorithm of log(n) complexity exists.
    """
    original_size = None
    if x.dim() > 2:
        original_size = x.size()
        x = x.view(x.size(0), -1)
    ax = x.data.abs().sum(0).view(-1)
    topk, ids = ax.topk(x.size(-1) - k, dim=0, largest=False)
    y = x.clone()
    for id in ids:
        y[:, (id)] = 0
    if original_size:
        y = y.view(original_size)
    return y


def topk(x, k):
    """ Proof-of-concept implementation of topk.
    """
    original_size = None
    if x.dim() > 2:
        original_size = x.size()
        x = x.view(x.size(0), -1)
    ax = torch.abs(x.data)
    topk, _ = ax.topk(k)
    topk = topk[:, (-1)]
    y = x.clone()
    y[ax < topk.repeat(x.size(-1), 1).transpose(0, 1)] = 0
    if original_size:
        y = y.view(original_size)
    return y


def sparsify_grad(v, k, simplified=True):
    if simplified:
        v.register_hook(lambda g: simplified_topk(g, k))
    else:
        v.register_hook(lambda g: topk(g, k))
    return v


class meLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, k=1,
        simplified=True):
        super(meLinear, self).__init__(in_features, out_features, bias)
        self.k = k
        self.simplified = simplified

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        return sparsify_grad(output, self.k, self.simplified)

    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class meConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, k=1, simplified=True):
        super(meConv2d, self).__init__(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias)
        self.k = k
        self.simplified = simplified

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)
        return sparsify_grad(output, self.k, self.simplified)

    def reset_parameters(self):
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1.0 / stdv


def clip_grad(v, min, max):
    v_tmp = v.expand_as(v)
    v_tmp.register_hook(lambda g: g.clamp(min, max))
    return v_tmp


class meLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None,
        k=1, simplified=False):
        super(meLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.k = k
        self.simplified = simplified
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size)
            )
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size,
            hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self
            .weight_hh)
        pre = sparsify_grad(pre, self.k, self.simplified)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c


class meLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
        return_sequences=True, grad_clip=None, k=1, simplified=False):
        super(meLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip
        self.k = k
        self.simplified = simplified
        kwargs = {'input_size': input_size, 'hidden_size': hidden_size,
            'bias': bias, 'grad_clip': grad_clip, 'k': k, 'simplified':
            simplified}
        self.cell0 = meLSTMCell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = hidden_size
            cell = meLSTMCell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            initial_states = [(zeros, zeros)] * self.num_layers
        assert len(initial_states) == self.num_layers
        states = initial_states
        outputs = []
        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, (t), :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx
                x = hx[0]
            outputs.append(hx)
        if self.return_sequences:
            hs, cs = zip(*outputs)
            h = torch.stack(hs).transpose(0, 1)
            c = torch.stack(cs).transpose(0, 1)
            output = h, c
        else:
            output = outputs[-1]
        return output


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out, use_cuda=torch.cuda.is_available())
        out = self.fc2(out)
        return out


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class GRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh_rz = Parameter(torch.Tensor(2 * hidden_size,
            hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        ih = F.linear(input, self.weight_ih, self.bias)
        hh_rz = F.linear(h, self.weight_hh_rz)
        if self.grad_clip:
            ih = clip_grad(ih, -self.grad_clip, self.grad_clip)
            hh_rz = clip_grad(hh_rz, -self.grad_clip, self.grad_clip)
        r = F.sigmoid(ih[:, :self.hidden_size] + hh_rz[:, :self.hidden_size])
        i = F.sigmoid(ih[:, self.hidden_size:self.hidden_size * 2] + hh_rz[
            :, self.hidden_size:])
        hhr = F.linear(h * r, self.weight_hh)
        if self.grad_clip:
            hhr = clip_grad(hhr, -self.grad_clip, self.grad_clip)
        n = F.relu(ih[:, self.hidden_size * 2:] + hhr)
        h = (1 - i) * n + i * h
        return h


class IndRNNCell(RNNCellBase):
    """
    References:
    Li et al. [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831).
    """

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(IndRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias
            ) + h * self.weight_hh
        if self.grad_clip:
            output = clip_grad(output, -self.grad_clip, self.grad_clip)
        output = F.relu(output)
        return output


class LSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self
            .weight_hh)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c


def cumax(logits, dim=-1):
    return torch.cumsum(F.softmax(logits, dim), dim=dim)


class LSTMONCell(RNNCellBase):
    """
    Shen & Tan et al. ORDERED NEURONS: INTEGRATING TREE STRUCTURES INTO RECURRENT NEURAL NETWORKS
    """

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(LSTMONCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(6 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(6 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(6 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self
            .weight_hh)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:self.hidden_size * 4])
        ff = cumax(pre[:, self.hidden_size * 4:self.hidden_size * 5])
        ii = 1 - cumax(pre[:, self.hidden_size * 5:self.hidden_size * 6])
        w = ff * ii
        f = f * w + (ff - w)
        i = i * w + (ii - w)
        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c


class LSTMPCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, recurrent_size, bias=True,
        grad_clip=None):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size,
            recurrent_size))
        self.weight_rec = Parameter(torch.Tensor(recurrent_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        pre = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self
            .weight_hh)
        if self.grad_clip:
            pre = clip_grad(pre, -self.grad_clip, self.grad_clip)
        i = F.sigmoid(pre[:, :self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size:self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2:self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3:])
        c = f * c + i * g
        h = o * F.tanh(c)
        h = F.linear(h, self.weight_rec)
        return h, c


class RNNCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih, self.bias) + F.linear(h,
            self.weight_hh)
        if self.grad_clip:
            output = clip_grad(output, -self.grad_clip, self.grad_clip)
        output = F.relu(output)
        return output


class RNNBase(Module):

    def __init__(self, mode, input_size, hidden_size, recurrent_size=None,
        num_layers=1, bias=True, return_sequences=True, grad_clip=None):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.grad_clip = grad_clip
        mode2cell = {'RNN': RNNCell, 'IndRNN': IndRNNCell, 'GRU': GRUCell,
            'MGRU': GRUCell, 'LSTM': LSTMCell, 'LSTMON': LSTMONCell,
            'LSTMP': LSTMPCell}
        Cell = mode2cell[mode]
        kwargs = {'input_size': input_size, 'hidden_size': hidden_size,
            'bias': bias, 'grad_clip': grad_clip}
        if self.mode == 'LSTMP':
            kwargs['recurrent_size'] = recurrent_size
        self.cell0 = Cell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'
                ] = recurrent_size if self.mode == 'LSTMP' else hidden_size
            cell = Cell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            if self.mode == 'LSTM' or self.mode == 'LSTMON':
                initial_states = [(zeros, zeros)] * self.num_layers
            elif self.mode == 'LSTMP':
                zeros_h = Variable(torch.zeros(input.size(0), self.
                    recurrent_size))
                initial_states = [(zeros_h, zeros)] * self.num_layers
            else:
                initial_states = [zeros] * self.num_layers
        assert len(initial_states) == self.num_layers
        states = initial_states
        outputs = []
        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, (t), :]
            for l in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(l))(x, states[l])
                states[l] = hx
                if self.mode.startswith('LSTM'):
                    x = hx[0]
                else:
                    x = hx
            outputs.append(hx)
        if self.return_sequences:
            if self.mode.startswith('LSTM'):
                hs, cs = zip(*outputs)
                h = torch.stack(hs).transpose(0, 1)
                c = torch.stack(cs).transpose(0, 1)
                output = h, c
            else:
                output = torch.stack(outputs).transpose(0, 1)
        else:
            output = outputs[-1]
        return output


class GRU(RNNBase):

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = GRU(input_size, hidden_size, num_layers=num_layers, bias
            =bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)
            ) for _ in range(self.num_layers)]
        out = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class IndRNN(RNNBase):
    """
    References:
    Li et al. [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831).
    """

    def __init__(self, *args, **kwargs):
        super(IndRNN, self).__init__('IndRNN', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = IndRNN(input_size, hidden_size, num_layers=num_layers,
            bias=bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)
            ) for _ in range(self.num_layers)]
        out = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class LSTM(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = LSTM(input_size, hidden_size, num_layers=num_layers,
            bias=bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        zeros = Variable(torch.zeros(x.size(0), self.hidden_size))
        initial_states = [(zeros, zeros)] * self.num_layers
        out, _ = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class LSTMON(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTMON, self).__init__('LSTMON', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = LSTMON(input_size, hidden_size, num_layers=num_layers,
            bias=bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        zeros = Variable(torch.zeros(x.size(0), self.hidden_size))
        initial_states = [(zeros, zeros)] * self.num_layers
        out, _ = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class LSTMP(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTMP, self).__init__('LSTMP', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, recurrent_size, num_layers,
        num_classes, bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.rnn = LSTMP(input_size, hidden_size, recurrent_size,
            num_layers=num_layers, bias=bias, return_sequences=False,
            grad_clip=grad_clip)
        self.fc = nn.Linear(recurrent_size, num_classes, bias=bias)

    def forward(self, x):
        zeros_h = Variable(torch.zeros(x.size(0), self.recurrent_size))
        zeros_c = Variable(torch.zeros(x.size(0), self.hidden_size))
        initial_states = [(zeros_h, zeros_c)] * self.num_layers
        out, _ = self.rnn(x, initial_states=None)
        out = self.fc(out)
        return out


class MGRU(RNNBase):

    def __init__(self, *args, **kwargs):
        super(MGRU, self).__init__('MGRU', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = MGRU(input_size, hidden_size, num_layers=num_layers,
            bias=bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)
            ) for _ in range(self.num_layers)]
        out = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


class RNN(RNNBase):

    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN', *args, **kwargs)


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
        bias=True, grad_clip=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = RNN(input_size, hidden_size, num_layers=num_layers, bias
            =bias, return_sequences=False, grad_clip=grad_clip)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)
            ) for _ in range(self.num_layers)]
        out = self.rnn(x, initial_states)
        out = self.fc(out)
        return out


ratio = 3


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding
            =2), SEWrapper(16, ratio), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5,
            padding=2), SEWrapper(32, ratio), nn.BatchNorm2d(32), nn.
            MaxPool2d(2), nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SEWrapper(nn.Module):

    def __init__(self, channels, ratio=4):
        super(SEWrapper, self).__init__()
        self.linear = nn.Sequential(nn.Linear(channels, channels // ratio),
            nn.ReLU(), nn.Linear(channels // ratio, channels), nn.Sigmoid())

    def forward(self, input):
        sq = input.mean(-1).mean(-1)
        ex = self.linear(sq)
        return input * ex.unsqueeze(-1).unsqueeze(-1)


class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input):
        return 1.67653251702 * input * F.sigmoid(input)


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = SiLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_DingKe_pytorch_workplace(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BinaryConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BinaryLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BinaryTanh(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(DNI(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(GRUCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(IndRNN(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(IndRNNCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(Linear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(MLP(*[], **{'input_size': 4, 'hidden_size': 4, 'num_classes': 4}), [torch.rand([4, 4, 4])], {})

    def test_009(self):
        self._check(Net(*[], **{'input_size': 4, 'hidden_size': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(Net1(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    def test_011(self):
        self._check(Net2(*[], **{'input_size': 4, 'hidden_size': 4, 'num_classes': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(RNN(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(RNNCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_014(self):
        self._check(RNNModel(*[], **{'input_size': 4, 'hidden_size': 4, 'num_layers': 1, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(ReLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(SEWrapper(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(SiLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_018(self):
        self._check(meConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(meLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

