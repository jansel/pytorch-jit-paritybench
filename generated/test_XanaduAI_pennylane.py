import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
benchmark_revisions = _module
benchmark_utils = _module
bm_entangling_layers = _module
bm_iqp_circuit = _module
bm_mutable_complicated_params = _module
bm_mutable_rotations = _module
edit_on_github = _module
conf = _module
directives = _module
pennylane = _module
_device = _module
_qubit_device = _module
_queuing_context = _module
_version = _module
about = _module
beta = _module
plugins = _module
default_tensor = _module
default_tensor_tf = _module
numpy_ops = _module
circuit_drawer = _module
charsets = _module
grid = _module
representation_resolver = _module
circuit_graph = _module
collections = _module
apply = _module
dot = _module
map = _module
qnode_collection = _module
sum = _module
configuration = _module
init = _module
interfaces = _module
autograd = _module
tf = _module
io = _module
measure = _module
numpy = _module
fft = _module
linalg = _module
random = _module
tensor = _module
wrapper = _module
operation = _module
ops = _module
cv = _module
qubit = _module
optimize = _module
adagrad = _module
adam = _module
gradient_descent = _module
momentum = _module
nesterov_momentum = _module
qng = _module
rms_prop = _module
rotoselect = _module
rotosolve = _module
default_gaussian = _module
default_qubit = _module
default_qubit_tf = _module
tf_ops = _module
qnn = _module
cost = _module
keras = _module
qnodes = _module
base = _module
decorator = _module
device_jacobian = _module
jacobian = _module
passthru = _module
rev = _module
templates = _module
broadcast = _module
embeddings = _module
amplitude = _module
angle = _module
basis = _module
displacement = _module
iqp = _module
qaoa = _module
squeezing = _module
layers = _module
basic_entangler = _module
cv_neural_net = _module
simplified_two_design = _module
strongly_entangling = _module
state_preparations = _module
arbitrary_state_preparation = _module
mottonen = _module
subroutines = _module
arbitrary_unitary = _module
double_excitation_unitary = _module
interferometer = _module
single_excitation_unitary = _module
uccsd = _module
utils = _module
variable = _module
vqe = _module
vqe = _module
wires = _module
pennylane_qchem = _module
qchem = _module
setup = _module
tests = _module
conftest = _module
test_active_space = _module
test_convert_hamiltonian = _module
test_decompose_hamiltonian = _module
test_generate_hamiltonian = _module
test_hf_state = _module
test_meanfield_data = _module
test_read_structure = _module
test_sd_excitations = _module
test_default_tensor = _module
test_default_tensor_tf = _module
test_circuit_drawer = _module
test_grid = _module
test_representation_resolver = _module
test_circuit_graph = _module
test_circuit_graph_hash = _module
test_qasm = _module
conftest = _module
test_collections = _module
test_qnode_collection = _module
conftest = _module
gate_data = _module
test_autograd = _module
test_tf = _module
test_torch = _module
test_cv_ops = _module
test_qubit_ops = _module
test_default_gaussian = _module
test_default_qubit = _module
test_default_qubit_tf = _module
test_cost = _module
test_keras = _module
test_qnn_torch = _module
test_qnode_base = _module
test_qnode_cv = _module
test_qnode_decorator = _module
test_qnode_jacobian = _module
test_qnode_metric_tensor = _module
test_qnode_passthru = _module
test_qnode_qubit = _module
test_qnode_reversible = _module
test_broadcast = _module
test_decorator = _module
test_embeddings = _module
test_integration = _module
test_layers = _module
test_state_preparations = _module
test_subroutines = _module
test_templ_utils = _module
test_about = _module
test_classical_gradients = _module
test_configuration = _module
test_device = _module
test_hermitian_edge_cases = _module
test_init = _module
test_io = _module
test_measure = _module
test_numpy_wrapper = _module
test_observable = _module
test_operation = _module
test_optimize = _module
test_optimize_qng = _module
test_prob = _module
test_quantum_gradients = _module
test_qubit_device = _module
test_queuing_context = _module
test_tensor_measurements = _module
test_utils = _module
test_variable = _module
test_vqe = _module
test_wires = _module

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


import abc


import warnings


from itertools import product


import numpy as np


import copy


from collections.abc import Sequence


from collections import Iterable


import inspect


from functools import partial


import numbers


import torch


from torch.autograd.function import once_differentiable


import numpy as onp


import functools


import math


from collections.abc import Iterable


from typing import Callable


from typing import Optional


from functools import lru_cache


from torch.autograd import Variable


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (function): a valid Python function

    Returns:
        dict: dictionary containing the argument name and tuple
        (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {k: (idx, v.default) for idx, (k, v) in enumerate(signature.parameters.items()) if v.default is not inspect.Parameter.empty}


def args_to_numpy(args):
    """Converts all Torch tensors in a list to NumPy arrays

    Args:
        args (list): list containing QNode arguments, including Torch tensors

    Returns:
        list: returns the same list, with all Torch tensors converted to NumPy arrays
    """
    res = []
    for i in args:
        if isinstance(i, torch.Tensor):
            if i.is_cuda:
                res.append(i.cpu().detach().numpy())
            else:
                res.append(i.detach().numpy())
        else:
            res.append(i)
    res = [(i.tolist() if isinstance(i, np.ndarray) and not i.shape else i) for i in res]
    return res


def kwargs_to_numpy(kwargs):
    """Converts all Torch tensors in a dictionary to NumPy arrays

    Args:
        args (dict): dictionary containing QNode keyword arguments, including Torch tensors

    Returns:
        dict: returns the same dictionary, with all Torch tensors converted to NumPy arrays
    """
    res = {}
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            if val.is_cuda:
                res[key] = val.cpu().detach().numpy()
            else:
                res[key] = val.detach().numpy()
        else:
            res[key] = val
    res = {k: (v.tolist() if isinstance(v, np.ndarray) and not v.shape else v) for k, v in res.items()}
    return res


def unflatten_torch(flat, model):
    """Restores an arbitrary nested structure to a flattened Torch tensor.

    Args:
        flat (torch.Tensor): 1D tensor of items
        model (array, Iterable, Number): model nested structure

    Returns:
        Tuple[list[torch.Tensor], torch.Tensor]: tuple containing elements of ``flat`` arranged
        into the nested structure of model, as well as the unused elements of ``flat``.

    Raises:
        TypeError: if ``model`` contains an object of unsupported type
    """
    if isinstance(model, (numbers.Number, str)):
        return flat[0], flat[1:]
    if isinstance(model, (torch.Tensor, np.ndarray)):
        try:
            idx = model.numel()
        except AttributeError:
            idx = model.size
        res = flat[:idx].reshape(model.shape)
        return res, flat[idx:]
    if isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = unflatten_torch(flat, x)
            res.append(val)
        return res, flat
    raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def to_torch(qnode):
    """Function that accepts a :class:`~.QNode`, and returns a PyTorch-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        torch.autograd.Function: the QNode as a PyTorch autograd function
    """


    class _TorchQNode(torch.autograd.Function):
        """The TorchQNode"""

        @staticmethod
        def set_trainable(args):
            """Given input arguments to the TorchQNode, determine which arguments
            are trainable and which aren't.

            Currently, all arguments are assumed to be nondifferentiable by default,
            unless the ``torch.tensor`` attribute ``requires_grad`` is set to True.

            This method calls the underlying :meth:`set_trainable_args` method of the QNode.
            """
            trainable_args = set()
            for idx, arg in enumerate(args):
                if getattr(arg, 'requires_grad', False):
                    trainable_args.add(idx)
            qnode.set_trainable_args(trainable_args)

        @staticmethod
        def forward(ctx, input_kwargs, *input_):
            """Implements the forward pass QNode evaluation"""
            ctx.args = args_to_numpy(input_)
            ctx.kwargs = kwargs_to_numpy(input_kwargs)
            ctx.save_for_backward(*input_)
            _TorchQNode.set_trainable(input_)
            res = qnode(*ctx.args, **ctx.kwargs)
            if not isinstance(res, np.ndarray):
                res = np.array(res)
            for i in input_:
                if isinstance(i, torch.Tensor):
                    if i.is_cuda:
                        cuda_device = i.get_device()
                        return torch.as_tensor(torch.from_numpy(res), device=cuda_device)
            return torch.from_numpy(res)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            """Implements the backwards pass QNode vector-Jacobian product"""
            jacobian = qnode.jacobian(ctx.args, ctx.kwargs)
            jacobian = torch.as_tensor(jacobian, dtype=grad_output.dtype)
            vjp = torch.transpose(grad_output.view(-1, 1), 0, 1) @ jacobian
            vjp = vjp.flatten()
            grad_input_list = unflatten_torch(vjp, ctx.saved_tensors)[0]
            grad_input = []
            for i, j in zip(grad_input_list, ctx.saved_tensors):
                res = torch.as_tensor(i, dtype=j.dtype)
                if j.is_cuda:
                    cuda_device = j.get_device()
                    res = torch.as_tensor(res, device=cuda_device)
                grad_input.append(res)
            return (None,) + tuple(grad_input)


    class TorchQNode(partial):
        """Torch QNode"""

        @property
        def interface(self):
            """String representing the QNode interface"""
            return 'torch'

        def __str__(self):
            """String representation"""
            detail = "<QNode: device='{}', func={}, wires={}, interface={}>"
            return detail.format(qnode.device.short_name, qnode.func.__name__, qnode.num_wires, self.interface)

        def __repr__(self):
            """REPL representation"""
            return self.__str__()
        print_applied = qnode.print_applied
        jacobian = qnode.jacobian
        metric_tensor = qnode.metric_tensor
        draw = qnode.draw
        func = qnode.func
        set_trainable_args = qnode.set_trainable_args
        get_trainable_args = qnode.get_trainable_args
        arg_vars = property(lambda self: qnode.arg_vars)
        num_variables = property(lambda self: qnode.num_variables)
        par_to_grad_method = property(lambda self: qnode.par_to_grad_method)

    @TorchQNode
    def custom_apply(*args, **kwargs):
        """Custom apply wrapper, to allow passing kwargs to the TorchQNode"""
        keyword_sig = _get_default_args(qnode.func)
        keyword_defaults = {k: v[1] for k, v in keyword_sig.items()}
        keyword_values = {}
        keyword_values.update(keyword_defaults)
        keyword_values.update(kwargs)
        return _TorchQNode.apply(keyword_values, *args)
    return custom_apply


class TorchLayer(Module):
    """Converts a :func:`~.QNode` to a Torch layer.

    The result can be used within the ``torch.nn``
    `Sequential <https://pytorch.org/docs/stable/nn.html#sequential>`__ or
    `Module <https://pytorch.org/docs/stable/nn.html#module>`__ classes for
    creating quantum and hybrid models.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Torch layer
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        init_method (callable): a `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__
            function for initializing the QNode weights. If not specified, weights are randomly
            initialized using the uniform distribution over :math:`[0, 2 \\pi]`.

    **Example**

    First let's define the QNode that we want to convert into a Torch layer:

    .. code-block:: python

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights_0, weight_1):
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            qml.Rot(*weights_0, wires=0)
            qml.RY(weight_1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    The signature of the QNode **must** contain an ``inputs`` named argument for input data,
    with all other arguments to be treated as internal weights. We can then convert to a Torch
    layer with:

    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    The internal weights of the QNode are automatically initialized within the
    :class:`~.TorchLayer` and must have their shapes specified in a ``weight_shapes`` dictionary.
    It is then easy to combine with other neural network layers from the
    `torch.nn <https://pytorch.org/docs/stable/nn.html>`__ module and create a hybrid:

    >>> clayer = torch.nn.Linear(2, 2)
    >>> model = torch.nn.Sequential(qlayer, clayer)

    .. UsageDetails::

        **QNode signature**

        The QNode must have a signature that satisfies the following conditions:

        - Contain an ``inputs`` named argument for input data.
        - All other arguments must accept an array or tensor and are treated as internal
          weights of the QNode.
        - All other arguments must have no default value.
        - The ``inputs`` argument is permitted to have a default value provided the gradient with
          respect to ``inputs`` is not required.
        - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args``
          or ``**kwargs`` present in the signature.

        **Initializing weights**

        The optional ``init_method`` argument of :class:`~.TorchLayer` allows for the initialization
        method of the QNode weights to be specified. The function passed to the argument must be
        from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ module. For
        example, weights can be randomly initialized from the normal distribution by passing:

        .. code-block::

            init_method = torch.nn.init.normal_

        If ``init_method`` is not specified, weights are randomly initialized from the uniform
        distribution on the interval :math:`[0, 2 \\pi]`.

        **Full code example**

        The code block below shows how a circuit composed of templates from the
        :doc:`/code/qml_templates` module can be combined with classical
        `Linear <https://pytorch.org/docs/stable/nn.html#linear>`__ layers to learn
        the two-dimensional `moons <https://scikit-learn.org/stable/modules/generated/sklearn
        .datasets.make_moons.html>`__ dataset.

        .. code-block:: python

            import numpy as np
            import pennylane as qml
            import torch
            import sklearn.datasets

            n_qubits = 2
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

            weight_shapes = {"weights": (3, n_qubits, 3)}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
            clayer1 = torch.nn.Linear(2, 2)
            clayer2 = torch.nn.Linear(2, 2)
            softmax = torch.nn.Softmax(dim=1)
            model = torch.nn.Sequential(clayer1, qlayer, clayer2, softmax)

            samples = 100
            x, y = sklearn.datasets.make_moons(samples)
            y_hot = np.zeros((samples, 2))
            y_hot[np.arange(samples), y] = 1

            X = torch.tensor(x).float()
            Y = torch.tensor(y_hot).float()

            opt = torch.optim.SGD(model.parameters(), lr=0.5)
            loss = torch.nn.L1Loss()

        The model can be trained using:

        .. code-block:: python

            epochs = 8
            batch_size = 5
            batches = samples // batch_size

            data_loader = torch.utils.data.DataLoader(list(zip(X, Y)), batch_size=batch_size,
                                                      shuffle=True, drop_last=True)

            for epoch in range(epochs):

                running_loss = 0

                for x, y in data_loader:
                    opt.zero_grad()

                    loss_evaluated = loss(model(x), y)
                    loss_evaluated.backward()

                    opt.step()

                    running_loss += loss_evaluated

                avg_loss = running_loss / batches
                print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

        An example output is shown below:

        .. code-block:: rst

            Average loss over epoch 1: 0.5089
            Average loss over epoch 2: 0.4765
            Average loss over epoch 3: 0.2710
            Average loss over epoch 4: 0.1865
            Average loss over epoch 5: 0.1670
            Average loss over epoch 6: 0.1635
            Average loss over epoch 7: 0.1528
            Average loss over epoch 8: 0.1528
    """

    def __init__(self, qnode, weight_shapes: dict, init_method: Optional[Callable]=None):
        if not TORCH_IMPORTED:
            raise ImportError('TorchLayer requires PyTorch. PyTorch can be installed using:\npip install torch\nAlternatively, visit https://pytorch.org/get-started/locally/ for detailed instructions.')
        super().__init__()
        self.sig = qnode.func.sig
        if self.input_arg not in self.sig:
            raise TypeError('QNode must include an argument with name {} for inputting data'.format(self.input_arg))
        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError('{} argument should not have its dimension specified in weight_shapes'.format(self.input_arg))
        if set(weight_shapes.keys()) | {self.input_arg} != set(self.sig.keys()):
            raise ValueError('Must specify a shape for every non-input parameter in the QNode')
        if qnode.func.var_pos:
            raise TypeError('Cannot have a variable number of positional arguments')
        if qnode.func.var_keyword:
            raise TypeError('Cannot have a variable number of keyword arguments')
        self.qnode = to_torch(qnode)
        weight_shapes = {weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ()) for weight, size in weight_shapes.items()}
        defaults = {name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty}
        self.input_is_default = self.input_arg in defaults
        if defaults - {self.input_arg} != set():
            raise TypeError('Only the argument {} is permitted to have a default'.format(self.input_arg))
        if not init_method:
            init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
        self.qnode_weights = {}
        for name, size in weight_shapes.items():
            if len(size) == 0:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(1))[0])
            else:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(*size)))
            self.register_parameter(name, self.qnode_weights[name])

    def forward(self, inputs):
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        if len(inputs.shape) == 1:
            return self._evaluate_qnode(inputs)
        return torch.stack([self._evaluate_qnode(x) for x in inputs])

    def _evaluate_qnode(self, x):
        """Evaluates the QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        qnode = self.qnode
        for arg in self.sig:
            if arg is not self.input_arg:
                w = self.qnode_weights[arg]
                qnode = functools.partial(qnode, w)
            elif self.input_is_default:
                qnode = functools.partial(qnode, **{self.input_arg: x})
            else:
                qnode = functools.partial(qnode, x)
        return qnode().type(x.dtype)

    def __str__(self):
        detail = '<Quantum Torch Layer: func={}>'
        return detail.format(self.qnode.func.__name__)
    __repr__ = __str__
    _input_arg = 'inputs'

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg

