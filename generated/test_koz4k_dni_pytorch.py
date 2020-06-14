import sys
_module = sys.modules[__name__]
del sys
dni = _module
main = _module
main = _module
main = _module
data = _module
generate = _module
main = _module
model = _module
setup = _module

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


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn import init


from functools import partial


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import time


import math


def _ones_like(tensor):
    return tensor.new().resize_(tensor.size()).fill_(1)


class _Manager:
    defer_backward = False
    deferred_gradients = []
    context_stack = []

    @classmethod
    def reset_defer_backward(cls):
        cls.defer_backward = False
        cls.deferred_gradients = []

    @classmethod
    def backward(cls, variable, gradient=None):
        if gradient is None:
            gradient = _ones_like(variable.data)
        if cls.defer_backward:
            cls.deferred_gradients.append((variable, gradient))
        else:
            variable.backward(gradient)

    @classmethod
    def get_current_context(cls):
        if cls.context_stack:
            return cls.context_stack[-1]
        else:
            return None


class UnidirectionalInterface(torch.nn.Module):
    """Basic `Interface` for unidirectional communication.

    Can be used to manually pass `messages` with methods `send` and `receive`.

    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    """

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer

    def receive(self, trigger):
        """Synthesizes a `message` based on `trigger`.

        Detaches `message` so no gradient will go through it during the
        backward pass.

        Args:
            trigger: `trigger` to use to synthesize a `message`.

        Returns:
            The synthesized `message`.
        """
        return self.synthesizer(trigger, _Manager.get_current_context()
            ).detach()

    def send(self, message, trigger):
        """Updates the estimate of synthetic `message` based on `trigger`.

        Synthesizes a `message` based on `trigger`, computes the MSE between it
        and the input `message` and backpropagates it to compute its gradient
        w.r.t. `Synthesizer` parameters. Does not backpropagate through
        `trigger`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        synthetic_message = self.synthesizer(trigger.detach(), _Manager.
            get_current_context())
        loss = F.mse_loss(synthetic_message, message.detach())
        _Manager.backward(loss)


class _SyntheticGradientUpdater(torch.autograd.Function):

    @staticmethod
    def forward(ctx, trigger, synthetic_gradient):
        _, needs_synthetic_gradient_grad = ctx.needs_input_grad
        if not needs_synthetic_gradient_grad:
            raise ValueError(
                'synthetic_gradient should need gradient but it does not')
        ctx.save_for_backward(synthetic_gradient)
        return trigger.clone()

    @staticmethod
    def backward(ctx, true_gradient):
        synthetic_gradient, = ctx.saved_variables
        batch_size, *_ = synthetic_gradient.size()
        grad_synthetic_gradient = 2 / batch_size * (synthetic_gradient -
            true_gradient)
        return true_gradient, grad_synthetic_gradient


class BackwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing gradients in the backward pass.

    Can be used to achieve an update unlock.

    Args:
        synthesizer: `Synthesizer` to use to generate gradients.
    """

    def forward(self, trigger):
        """Normal forward pass, synthetic backward pass.

        Convenience method combining `backward` and `make_trigger`. Can be
        used when we want to backpropagate synthetic gradients from and
        intercept real gradients at the same `Variable`, for example for
        update decoupling feed-forward networks.

        Backpropagates synthetic gradient from `trigger` and returns a copy of
        `trigger` with a synthetic gradient update operation attached.

        Works only in `training` mode, otherwise just returns the input
        `trigger`.

        Args:
            trigger: `trigger` to backpropagate synthetic gradient from and
                intercept real gradient at.

        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            self.backward(trigger)
            return self.make_trigger(trigger.detach())
        else:
            return trigger

    def backward(self, trigger, factor=1):
        """Backpropagates synthetic gradient from `trigger`.

        Computes synthetic gradient based on `trigger`, scales it by `factor`
        and backpropagates it from `trigger`.

        Works only in `training` mode, otherwise is a no-op.

        Args:
            trigger: `trigger` to compute synthetic gradient based on and to
                backpropagate it from.
            factor (optional): Factor by which to scale the synthetic gradient.
                Defaults to 1.
        """
        if self.training:
            synthetic_gradient = self.receive(trigger)
            _Manager.backward(trigger, synthetic_gradient.data * factor)

    def make_trigger(self, trigger):
        """Attaches a synthetic gradient update operation to `trigger`.

        Returns a `Variable` with the same `data` as `trigger`, that during
        the backward pass will intercept gradient passing through it and use
        this gradient to update the `Synthesizer`'s estimate.

        Works only in `training` mode, otherwise just returns the input
        `trigger`.

        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            return _SyntheticGradientUpdater.apply(trigger, self.
                synthesizer(trigger, _Manager.get_current_context()))
        else:
            return trigger


class ForwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing activations in the forward pass.

    Can be used to achieve a forward unlock. It does not make too much sense to
    use it on its own, as it breaks backpropagation (no gradients pass through
    `ForwardInterface`). To achieve both forward and update unlock, use
    `BidirectionalInterface`.

    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    """

    def forward(self, message, trigger):
        """Synthetic forward pass, no backward pass.

        Convenience method combining `send` and `receive`. Updates the
        `message` estimate based on `trigger` and returns a synthetic
        `message`.

        Works only in `training` mode, otherwise just returns the input
        `message`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.

        Returns:
            The synthesized `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message


class BidirectionalInterface(torch.nn.Module):
    """`Interface` for synthesizing both activations and gradients w.r.t. them.

    Can be used to achieve a full unlock.

    Args:
        forward_synthesizer: `Synthesizer` to use to generate `messages`.
        backward_synthesizer: `Synthesizer` to use to generate gradients w.r.t.
            `messages`.
    """

    def __init__(self, forward_synthesizer, backward_synthesizer):
        super().__init__()
        self.forward_interface = ForwardInterface(forward_synthesizer)
        self.backward_interface = BackwardInterface(backward_synthesizer)

    def forward(self, message, trigger):
        """Synthetic forward pass, synthetic backward pass.

        Convenience method combining `send` and `receive`. Can be used when we
        want to `send` and immediately `receive` using the same `trigger`. For
        more complex scenarios, `send` and `receive` need to be used
        separately.

        Updates the `message` estimate based on `trigger`, backpropagates
        synthetic gradient from `message` and returns a synthetic `message`
        with a synthetic gradient update operation attached.

        Works only in `training` mode, otherwise just returns the input
        `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message

    def receive(self, trigger):
        """Combination of `ForwardInterface.receive` and
        `BackwardInterface.make_trigger`.

        Generates a synthetic `message` based on `trigger` and attaches to it
        a synthetic gradient update operation.

        Args:
            trigger: `trigger` to use to synthesize a `message`.

        Returns:
            The synthesized `message` with a synthetic gradient update
            operation attached.
        """
        message = self.forward_interface.receive(trigger)
        return self.backward_interface.make_trigger(message)

    def send(self, message, trigger):
        """Combination of `ForwardInterface.send` and
        `BackwardInterface.backward`.

        Updates the estimate of synthetic `message` based on `trigger` and
        backpropagates synthetic gradient from `message`.

        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger` and that synthetic gradient should be backpropagated
                from.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        self.forward_interface.send(message, trigger)
        self.backward_interface.backward(message)


class BasicSynthesizer(torch.nn.Module):
    """Basic `Synthesizer` based on an MLP with ReLU activation.

    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim, n_hidden=0, hidden_dim=None, trigger_dim
        =None, context_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        if trigger_dim is None:
            trigger_dim = output_dim
        top_layer_dim = output_dim if n_hidden == 0 else hidden_dim
        self.input_trigger = torch.nn.Linear(in_features=trigger_dim,
            out_features=top_layer_dim)
        if context_dim is not None:
            self.input_context = torch.nn.Linear(in_features=context_dim,
                out_features=top_layer_dim)
        else:
            self.input_context = None
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=
            hidden_dim, out_features=hidden_dim if layer_index < n_hidden -
            1 else output_dim) for layer_index in range(n_hidden)])
        if n_hidden > 0:
            init.constant(self.layers[-1].weight, 0)
        else:
            init.constant(self.input_trigger.weight, 0)
            if context_dim is not None:
                init.constant(self.input_context.weight, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.

        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).

        Returns:
            The synthesized `message`.
        """
        last = self.input_trigger(trigger)
        if self.input_context is not None:
            last += self.input_context(context)
        for layer in self.layers:
            last = layer(F.relu(last))
        return last


_global_config['cuda'] = 4


def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    if args.cuda:
        result = result.cuda()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(dim=indexes_rank, index=indexes.data.unsqueeze(dim=
        indexes_rank), value=1)
    return Variable(result)


_global_config['context'] = 4


_global_config['dni'] = 4


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        if args.dni:
            self.backward_interface = dni.BackwardInterface(ConvSynthesizer())
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x, y=None):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2)
        if args.dni and self.training:
            if args.context:
                context = one_hot(y, 10)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2_bn(self.fc2(x))
        return F.log_softmax(x)


class ConvSynthesizer(nn.Module):

    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.input_context = nn.Linear(10, 20)
        self.hidden = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.output = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        nn.init.constant(self.output.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        if context is not None:
            x += self.input_context(context).unsqueeze(2).unsqueeze(3
                ).expand_as(x)
        x = self.hidden(F.relu(x))
        return self.output(F.relu(x))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(784, 256, bias=False)
        self.hidden1_bn = nn.BatchNorm1d(256)
        self.hidden2 = nn.Linear(256, 256, bias=False)
        self.hidden2_bn = nn.BatchNorm1d(256)
        if args.dni:
            if args.context:
                context_dim = 10
            else:
                context_dim = None
            self.bidirectional_interface = dni.BidirectionalInterface(dni.
                BasicSynthesizer(output_dim=256, n_hidden=2, trigger_dim=
                784, context_dim=context_dim), dni.BasicSynthesizer(
                output_dim=256, n_hidden=2, context_dim=context_dim))
        self.output = nn.Linear(256, 10, bias=False)
        self.output_bn = nn.BatchNorm1d(10)

    def forward(self, x, y=None):
        input_flat = x.view(x.size()[0], -1)
        x = self.hidden1_bn(self.hidden1(input_flat))
        x = self.hidden2_bn(self.hidden2(F.relu(x)))
        if args.dni and self.training:
            if args.context:
                context = one_hot(y, 10)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.bidirectional_interface(x, input_flat)
        x = self.output_bn(self.output(F.relu(x)))
        return F.log_softmax(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(784, 256, bias=False)
        self.hidden1_bn = nn.BatchNorm1d(256)
        self.hidden2 = nn.Linear(256, 256, bias=False)
        self.hidden2_bn = nn.BatchNorm1d(256)
        if args.dni:
            if args.context:
                context_dim = 10
            else:
                context_dim = None
            self.backward_interface = dni.BackwardInterface(dni.
                BasicSynthesizer(output_dim=256, n_hidden=1, context_dim=
                context_dim))
        self.output = nn.Linear(256, 10, bias=False)
        self.output_bn = nn.BatchNorm1d(10)

    def forward(self, x, y=None):
        x = x.view(x.size()[0], -1)
        x = self.hidden1_bn(self.hidden1(x))
        x = self.hidden2_bn(self.hidden2(F.relu(x)))
        if args.dni and self.training:
            if args.context:
                context = one_hot(y, 10)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
        x = self.output_bn(self.output(F.relu(x)))
        return F.log_softmax(x)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
        tie_weights=False, use_dni=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=
                dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[
                    rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                    )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=
                nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        if use_dni:
            if rnn_type == 'LSTM':
                output_dim = 2 * nhid
            else:
                output_dim = nhid
            self.backward_interface = dni.BackwardInterface(dni.
                BasicSynthesizer(output_dim, n_hidden=2))
        else:
            self.backward_interface = None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def join_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            hidden = torch.cat(hidden, dim=2)
        return hidden

    def split_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            h, c = hidden.chunk(2, dim=2)
            hidden = h.contiguous(), c.contiguous()
        return hidden

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        if self.backward_interface is not None:
            hidden = self.join_hidden(hidden)
            hidden = self.backward_interface.make_trigger(hidden)
            hidden = self.split_hidden(hidden)
        output, hidden = self.rnn(emb, hidden)
        if self.backward_interface is not None:
            hidden = self.join_hidden(hidden)
            self.backward_interface.backward(hidden, factor=0.1)
            hidden = self.split_hidden(hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
            output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_koz4k_dni_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BackwardInterface(*[], **{'synthesizer': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicSynthesizer(*[], **{'output_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BidirectionalInterface(*[], **{'forward_synthesizer': 4, 'backward_synthesizer': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ForwardInterface(*[], **{'synthesizer': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Net(*[], **{}), [torch.rand([784, 784])], {})

