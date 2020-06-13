import sys
_module = sys.modules[__name__]
del sys
LearningMachine = _module
Model = _module
ModelConf = _module
calculate_AUC = _module
get_results = _module
BaseLayer = _module
BiGRU = _module
BiGRULast = _module
BiLSTM = _module
BiLSTMAtt = _module
BiLSTMLast = _module
BiQRNN = _module
CRF = _module
Conv = _module
Conv2D = _module
ConvPooling = _module
Dropout = _module
Embedding = _module
EncoderDecoder = _module
HighwayLinear = _module
Linear = _module
Pooling = _module
Pooling1D = _module
Pooling2D = _module
Transformer = _module
block_zoo = _module
Attention = _module
BiAttFlow = _module
BilinearAttention = _module
FullAttention = _module
Interaction = _module
LinearAttention = _module
MatchAttention = _module
Seq2SeqAttention = _module
attentions = _module
CNNCharEmbedding = _module
LSTMCharEmbedding = _module
embedding = _module
SLUDecoder = _module
SLUEncoder = _module
encoder_decoder = _module
Add2D = _module
Add3D = _module
ElementWisedMultiply2D = _module
ElementWisedMultiply3D = _module
MatrixMultiply = _module
Minus2D = _module
Minus3D = _module
math = _module
LayerNorm = _module
normalizations = _module
CalculateDistance = _module
Combination = _module
Concat2D = _module
Concat3D = _module
Expand_plus = _module
Flatten = _module
Match = _module
op = _module
MLP = _module
MultiHeadAttention = _module
transformer = _module
CellDict = _module
ChineseTokenizer = _module
EnglishPOSTagger = _module
EnglishTextPreprocessor = _module
EnglishTokenizer = _module
LRScheduler = _module
Stopwords = _module
StreamingRecorder = _module
core = _module
get_20_newsgroups = _module
get_QNLI = _module
get_QQP = _module
get_WikiQACorpus = _module
BaseLossConf = _module
CRFLoss = _module
FocalLoss = _module
Loss = _module
losses = _module
Evaluator = _module
metrics = _module
conlleval = _module
slot_tagging_metrics = _module
get_model_graph = _module
main = _module
mv = _module
optimizers = _module
predict = _module
preparation = _module
problem = _module
register_block = _module
settings = _module
test = _module
calculate_auc = _module
tagging_schemes_converter = _module
train = _module
BPEEncoder = _module
DocInherit = _module
ProcessorsScheduler = _module
utils = _module
common_utils = _module
corpus_utils = _module
exceptions = _module
philly_utils = _module

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


import random


import logging


import copy


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


from copy import deepcopy


import torch.autograd as autograd


from collections import OrderedDict


from torch import autograd


from torch.nn.parameter import Parameter


from torch.autograd import Variable


import math


from torch.nn import CrossEntropyLoss


from torch.nn import L1Loss


from torch.nn import MSELoss


from torch.nn import NLLLoss


from torch.nn import PoissonNLLLoss


from torch.nn import NLLLoss2d


from torch.nn import KLDivLoss


from torch.nn import BCELoss


from torch.nn import BCEWithLogitsLoss


from torch.nn import MarginRankingLoss


from torch.nn import HingeEmbeddingLoss


from torch.nn import MultiLabelMarginLoss


from torch.nn import SmoothL1Loss


from torch.nn import SoftMarginLoss


from torch.nn import MultiLabelSoftMarginLoss


from torch.nn import CosineEmbeddingLoss


from torch.nn import MultiMarginLoss


from torch.nn import TripletMarginLoss


EMBED_LAYER_NAME = 'Embedding'


class BaseError(RuntimeError):
    """ Error base class

    """

    def __init__(self, arg, err_id=None):
        self.arg = arg
        self.err_id = err_id

    def __str__(self):
        if self.err_id is None:
            return self.arg
        else:
            return 'error=%d, %s' % (self.err_id, self.arg)


class LayerConfigUndefinedError(BaseError):
    """ Errors occur when the corresponding configuration class of a layer is not defined

    """
    pass


EMBED_LAYER_ID = 'embedding'


class ConfigurationError(BaseError):
    """ Errors occur when model configuration

    """
    pass


def get_conf(layer_id, layer_name, input_layer_ids, all_layer_configs,
    model_input_ids, use_gpu, conf_dict=None, shared_conf=None,
    succeed_embedding_flag=False, output_layer_flag=False, target_num=None,
    fixed_lengths=None, target_dict=None):
    """ get layer configuration

    Args
        layer_id: layer identifier
        layer_name: name of layer such as BiLSTM
        input_layer_ids (list): the inputs of current layer
        all_layer_configs (dict): records the conf class of each layer.
        model_input_ids (set): the inputs of the model, e.g. ['query', 'passage']
        use_gpu:
        conf_dict:
        shared_conf: if fixed_lengths is not None, the output_dim of shared_conf should be corrected!
        flag:
        output_layer_flag:
        target_num: used for inference the dimension of output space if someone declare a dimension of -1
        fixed_lengths
    Returns:
        configuration class coresponds to the layer

    """
    if shared_conf:
        conf = copy.deepcopy(shared_conf)
    else:
        try:
            conf_dict['use_gpu'] = use_gpu
            if layer_id == EMBED_LAYER_ID:
                conf_dict['weight_on_gpu'] = conf_dict['conf']['weight_on_gpu']
                del conf_dict['conf']['weight_on_gpu']
            if layer_name == 'Linear':
                if isinstance(conf_dict['hidden_dim'], list):
                    if conf_dict['hidden_dim'][-1] == -1:
                        assert output_layer_flag is True, 'Only in the last layer, hidden_dim == -1 is allowed!'
                        assert target_num is not None, 'Number of targets should be given!'
                        conf_dict['hidden_dim'][-1] = target_num
                    elif conf_dict['hidden_dim'][-1] == '#target#':
                        logging.info(
                            '#target# position will be replace by target num: %d'
                             % target_num)
                        conf_dict['hidden_dim'][-1] = target_num
                elif isinstance(conf_dict['hidden_dim'], int) and conf_dict[
                    'hidden_dim'] == -1:
                    assert output_layer_flag is True, 'Only in the last layer, hidden_dim == -1 is allowed!'
                    assert target_num is not None, 'Number of targets should be given!'
                    conf_dict['hidden_dim'] = target_num
                elif isinstance(conf_dict['hidden_dim'], str) and conf_dict[
                    'hidden_dim'] == '#target#':
                    logging.info(
                        '#target# position will be replace by target num: %d' %
                        target_num)
                    conf_dict['hidden_dim'] = target_num
            if layer_name == 'CRF':
                conf_dict['target_dict'] = target_dict
            conf = eval(layer_name + 'Conf')(**conf_dict)
        except NameError as e:
            raise LayerConfigUndefinedError('"%sConf" has not been defined' %
                layer_name)
    if layer_name == EMBED_LAYER_NAME:
        pass
    else:
        for input_layer_id in input_layer_ids:
            if not (input_layer_id in all_layer_configs or input_layer_id in
                model_input_ids):
                raise ConfigurationError(
                    'The input %s of layer %s does not exist. Please define it before defining layer %s!'
                     % (input_layer_id, layer_id, layer_id))
        former_output_ranks = [(all_layer_configs[input_layer_id].
            output_rank if input_layer_id in all_layer_configs else
            all_layer_configs[EMBED_LAYER_ID].output_rank) for
            input_layer_id in input_layer_ids]
        conf.input_dims = [(all_layer_configs[input_layer_id].output_dim if
            input_layer_id in all_layer_configs else all_layer_configs[
            EMBED_LAYER_ID].output_dim) for input_layer_id in input_layer_ids]
        if len(input_layer_ids) == 1 and input_layer_ids[0
            ] in model_input_ids and fixed_lengths:
            conf.input_dims[0][1] = fixed_lengths[input_layer_ids[0]]
        if conf.num_of_inputs > 0:
            if conf.num_of_inputs != len(input_layer_ids):
                raise ConfigurationError(
                    '%s only accept %d inputs but you feed %d inputs to it!' %
                    (layer_name, conf.num_of_inputs, len(input_layer_ids)))
        elif conf.num_of_inputs == -1:
            conf.num_of_inputs = len(input_layer_ids)
            if isinstance(conf.input_ranks, list):
                conf.input_ranks = conf.input_ranks * conf.num_of_inputs
            else:
                logging.warning(
                    '[For developer of %s] The input_ranks attribute should be a list!'
                     % layer_name)
                [conf.input_ranks] * conf.num_of_inputs
        for input_rank, former_output_rank in zip(conf.input_ranks,
            former_output_ranks):
            if input_rank != -1 and input_rank != former_output_rank:
                raise ConfigurationError(
                    'Input ranks of %s are inconsistent with former layers' %
                    layer_id)
        conf.input_ranks = copy.deepcopy(former_output_ranks)
    conf.inference()
    conf.verify()
    former_conf = None if len(all_layer_configs) == 0 else list(
        all_layer_configs.values())[-1]
    conf.verify_former_block(former_conf)
    logging.debug(
        'Layer id: %s; name: %s; input_dims: %s; input_ranks: %s; output_dim: %s; output_rank: %s'
         % (layer_id, layer_name, conf.input_dims if layer_id !=
        'embedding' else 'None', conf.input_ranks, conf.output_dim, conf.
        output_rank))
    return conf


def get_layer(layer_name, conf):
    """

    Args:
        layer_name:
        conf:  configuration class

    Returns:
        specific layer

    """
    try:
        layer = eval(layer_name)(conf)
    except NameError as e:
        raise Exception('%s; Layer "%s" has not been defined' % (str(e),
            layer_name))
    return layer


class BaseLayer(nn.Module):
    """The base class of layers

    Args:
        layer_conf (BaseConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BaseLayer, self).__init__()
        self.layer_conf = layer_conf

    def forward(self, *args):
        """

        Args:
            *args (list): a list of args in which arg should be a pair of (representation, length)

        Returns:
            None

        """
        pass

    def is_cuda(self):
        """ To judge if the layer is on CUDA
        if there are parameters in this layer, judge according to the parameters;
        else: judge according to the self.layer_conf.use_gpu

        Returns:
            bool: whether to use gpu

        """
        try:
            ret = self.layer_conf.use_gpu
        except StopIteration as e:
            if not hasattr(self, 'layer_conf'):
                logging.error('Layer.layer_conf must be defined!')
            else:
                logging.error(e)
        return ret


class ForgetMult(torch.nn.Module):
    """ForgetMult computes a simple recurrent equation:
    h_t = f_t * x_t + (1 - f_t) * h_{t-1}

    This equation is equivalent to dynamic weighted averaging.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - F (seq_len, batch, input_size): tensor containing the forget gate values, assumed in range [0, 1].
        - hidden_init (batch, input_size): tensor containing the initial hidden state for the recurrence (h_{t-1}).
    """

    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None:
                h = h + (1 - forgets[i]) * prev_h
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)


class QRNNLayer(nn.Module):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (1, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False,
        zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()
        assert window in [1, 2
            ], 'This QRNN implementation currently only handles convolutional window of size 1 or size 2'
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.
            hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()
        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :
                ] * 0)
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            source = torch.cat([X, Xm1], 2)
        Y = self.linear(source)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        if self.zoneout:
            if self.training:
                mask = F.new_empty(F.size(), requires_grad=False).bernoulli_(
                    1 - self.zoneout)
                F = F * mask
            else:
                F *= 1 - self.zoneout
        C = ForgetMult()(F, Z, hidden)
        if self.output_gate:
            H = torch.sigmoid(O) * C
        else:
            H = C
        if self.window > 1 and self.save_prev_x:
            self.prevX = X[-1:, :, :].detach()
        return H, C[-1:, :, :]


class QRNN(torch.nn.Module):
    """Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        dropout: Whether to use dropout between QRNN layers. Default: 0.
        bidirectional: If True, becomes a bidirectional QRNN. Default: False.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size * num_directions): tensor containing the output of the QRNN for each timestep.
        - h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
        batch_first=False, dropout=0.0, bidirectional=False, **kwargs):
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'
        super(QRNN, self).__init__()
        if bidirectional:
            self.layers = torch.nn.ModuleList([QRNNLayer(input_size if l < 
                2 else hidden_size * 2, hidden_size, **kwargs) for l in
                range(num_layers * 2)])
        else:
            self.layers = torch.nn.ModuleList([QRNNLayer(input_size if l ==
                0 else hidden_size, hidden_size, **kwargs) for l in range(
                num_layers)])
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        assert len(self.layers) == self.num_layers * self.num_directions

    def tensor_reverse(self, tensor):
        return tensor.flip(0)

    def reset(self):
        """If your convolutional window is greater than 1, you must reset at the beginning of each new sequence"""
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []
        for i in range(self.num_layers):
            all_output = []
            for j in range(self.num_directions):
                l = i * self.num_directions + j
                layer = self.layers[l]
                if j == 1:
                    input = self.tensor_reverse(input)
                output, hn = layer(input, None if hidden is None else hidden[l]
                    )
                next_hidden.append(hn)
                if j == 1:
                    output = self.tensor_reverse(output)
                all_output.append(output)
            input = torch.cat(all_output, input.dim() - 1)
            if self.dropout != 0 and i < self.num_layers - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout,
                    training=self.training, inplace=False)
        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers * self
            .num_directions, *next_hidden[0].size()[-2:])
        return input, next_hidden


class Transformer(nn.Module):
    """ Transformer layer

    Args:
        layer_conf (TransformerConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Transformer, self).__init__()
        self.layer_conf = layer_conf
        self.transformer_layer = nn.ModuleList([copy.deepcopy(nn.ModuleList
            ([eval(layer_conf.attention_name)(layer_conf.attention_conf_cls
            ), eval(layer_conf.layernorm1_name)(layer_conf.
            layernorm1_conf_cls), eval(layer_conf.mlp_name)(layer_conf.
            mlp_conf_cls), eval(layer_conf.layernorm2_name)(layer_conf.
            layernorm2_conf_cls)])) for _ in range(self.layer_conf.n_layer)])

    def forward(self, string, string_len):
        """ process input

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]
        Returns:
            Tensor : [batch_size, seq_len, output_dim], [batch_size]
        """
        h = string
        l = string_len
        for block in self.transformer_layer:
            a, a_len = block[0](h, 1)
            n, n_len = block[1](a + h, a_len)
            m, m_len = block[2](n, n_len)
            h, l = block[3](m + n, m_len)
        return h, l


class Add2D(nn.Module):
    """ Add2D layer to get sum of two sequences(2D representation)

    Args:
        layer_conf (Add2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Add2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Add2D layer returns is the length of first input')

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, output_dim], [batch_size]
        """
        return torch.add(args[0], args[2]), args[1]


class Add3D(nn.Module):
    """ Add3D layer to get sum of two sequences(3D representation)

    Args:
        layer_conf (Add3DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Add3D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Add3D layer returns is the length of first input')

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        dim_flag = True
        input_dims = list(self.layer_conf.input_dims)
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2
            ].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] ==
                1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError(
                'For layer Add3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal'
                )
        return torch.add(args[0], args[2]), args[1]


class ElementWisedMultiply2D(nn.Module):
    """ ElementWisedMultiply2D layer to do Element-Wised Multiply of two sequences(2D representation)

    Args:
        layer_conf (ElementWisedMultiply2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(ElementWisedMultiply2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length ElementWisedMultiply2D layer returns is the length of first input'
            )

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]


        Returns:
            Tensor: [batch_size, output_dim], [batch_size]
        """
        return torch.addcmul(torch.zeros(args[0].size()).to('cuda'), 1,
            args[0], args[2]), args[1]


class ElementWisedMultiply3D(nn.Module):
    """ ElementWisedMultiply3D layer to do Element-Wised Multiply of two sequences(3D representation)

    Args:
        layer_conf (ElementWisedMultiply3DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(ElementWisedMultiply3D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length ElementWisedMultiply3D layer returns is the length of first input'
            )

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]


        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        dim_flag = True
        input_dims = list(self.layer_conf.input_dims)
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2
            ].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] ==
                1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError(
                'For layer ElementWisedMultiply3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal'
                )
        return torch.addcmul(torch.zeros(args[0].size()).to('cuda'), 1,
            args[0], args[2]), args[1]


class MatrixMultiply(nn.Module):
    """ MatrixMultiply layer to multiply two matrix

    Args:
        layer_conf (MatrixMultiplyConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(MatrixMultiply, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length MatrixMultiply layer returns is the length of first input'
            )

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        if self.layer_conf.operation == 'common':
            if args[0].shape[2] == args[2].shape[1]:
                return torch.matmul(args[0], args[2]), args[1]
            else:
                raise Exception(
                    'the dimensions of the two matrix for multiply is illegal')
        if self.layer_conf.operation == 'seq_based':
            if args[0].shape[1] == args[2].shape[1]:
                string = args[0].permute(0, 2, 1)
                return torch.matmul(string, args[2]), args[1]
            else:
                raise Exception(
                    'the dimensions of the two matrix for multiply is illegal')
        if self.layer_conf.operation == 'dim_based':
            if args[0].shape[2] == args[2].shape[2]:
                string = args[2].permute(0, 2, 1)
                return torch.matmul(args[0], string), args[1]
            else:
                raise Exception(
                    'the dimensions of the two matrix for multiply is illegal')


class Minus2D(nn.Module):
    """Minus2D layer to get subtraction of two sequences(2D representation)

    Args:
        layer_conf (Minus2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Minus2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Minus2D layer returns is the length of first input')

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, output_dim], [batch_size]

        """
        if self.layer_conf.abs_flag == False:
            return args[0] - args[2], args[1]
        if self.layer_conf.abs_flag == True:
            return torch.abs(args[0] - args[2]), args[1]


class Minus3D(nn.Module):
    """ Minus3D layer to get subtraction of two sequences(3D representation)

    Args:
        layer_conf (Minus3DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Minus3D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Minus3D layer returns is the length of first input')

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        dim_flag = True
        input_dims = list(self.layer_conf.input_dims)
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2
            ].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] ==
                1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError(
                'For layer Minus3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal'
                )
        if self.layer_conf.abs_flag == False:
            return args[0] - args[2], args[1]
        if self.layer_conf.abs_flag == True:
            return torch.abs(args[0] - args[2]), args[1]


class LayerNorm(nn.Module):
    """ LayerNorm layer

    Args:
        layer_conf (LayerNormConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(LayerNorm, self).__init__()
        self.layer_conf = layer_conf
        self.g = nn.Parameter(torch.ones(self.layer_conf.input_dims[0][-1]))
        self.b = nn.Parameter(torch.zeros(self.layer_conf.input_dims[0][-1]))
        self.e = 1e-05

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        u = string.mean(-1, keepdim=True)
        s = (string - u).pow(2).mean(-1, keepdim=True)
        string = (string - u) / torch.sqrt(s + self.e)
        return self.g * string + self.b, string_len


class Combination(nn.Module):
    """ Combination layer to merge the representation of two sequence

    Args:
        layer_conf (CombinationConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Combination, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Combination layer returns is the length of first input'
            )

    def forward(self, *args):
        """ process inputs

        Args:
            args (list): [string, string_len, string2, string2_len, ...]
                e.g. string (Variable): [batch_size, dim], string_len (ndarray): [batch_size]

        Returns:
            Variable: [batch_size, output_dim], None

        """
        result = []
        if 'origin' in self.layer_conf.operations:
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    result.append(input)
        if 'difference' in self.layer_conf.operations:
            result.append(torch.abs(args[0] - args[2]))
        if 'dot_multiply' in self.layer_conf.operations:
            result_multiply = None
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    if result_multiply is None:
                        result_multiply = input
                    else:
                        result_multiply = result_multiply * input
            result.append(result_multiply)
        last_dim = len(args[0].size()) - 1
        return torch.cat(result, last_dim), args[1]


class Concat2D(nn.Module):
    """ Concat2D layer to merge sum of sequences(2D representation)

    Args:
        layer_conf (Concat2DConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Concat2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Concat2D layer returns is the length of first input')

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len, ...
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, output_dim], [batch_size]

        """
        result = []
        for idx, input in enumerate(args):
            if idx % 2 == 0:
                result.append(input)
        return torch.cat(result, self.layer_conf.concat2D_axis), args[1]


class Concat3D(nn.Module):
    """ Concat3D layer to merge sum of sequences(3D representation)

    Args:
        layer_conf (Concat3DConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Concat3D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning(
            'The length Concat3D layer returns is the length of first input')

    def forward(self, *args):
        """ process inputs

        Args:
            *args: (Tensor): string, string_len, string2, string2_len, ...
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        result = []
        if self.layer_conf.concat3D_axis == 1:
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    result.append(input)
        if self.layer_conf.concat3D_axis == 2:
            input_shape = args[0].shape[1]
            for idx, input in enumerate(args):
                if idx % 2 == 0:
                    if input_shape == input.shape[1]:
                        result.append(input)
                    else:
                        raise Exception(
                            'Concat3D with axis = 2 require that the input sequences length should be the same!'
                            )
        return torch.cat(result, self.layer_conf.concat3D_axis), args[1]


class Flatten(nn.Module):
    """  Flatten layer to flatten the input from [bsatch_size, seq_len, dim] to [batch_size, seq_len*dim]

    Args:
        layer_conf(FlattenConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Flatten, self).__init__()
        self.layer_conf = layer_conf

    def forward(self, string, string_len):
        """ process input

        Args:
            *args: (Tensor): string,string_len
                e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]
        Returns:
            Tensor: [batch_size, seq_len*dim], [batch_size]
        """
        flattened = string.view(string.shape[0], -1)
        string_len = flattened.size(1)
        return flattened, string_len


class MLP(nn.Module):
    """ MLP layer

    Args:
        layer_conf (MLPConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(MLP, self).__init__()
        self.layer_conf = layer_conf
        self.n_state = self.layer_conf.input_dims[0][-1]
        self.c_fc = nn.Linear(self.layer_conf.input_dims[0][-1], 4 * self.
            n_state)
        self.c_proj = nn.Linear(4 * self.n_state, self.layer_conf.
            input_dims[0][-1])

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]

        """
        h = self.gelu(self.c_fc(string))
        h2 = self.c_proj(h)
        return nn.Dropout(self.layer_conf.dropout)(h2), string_len


class MultiHeadAttention(nn.Module):
    """ MultiHeadAttention Layer

    Args:
        layer_conf (MultiHeadAttentionConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(MultiHeadAttention, self).__init__()
        self.layer_conf = layer_conf
        self.split_size = self.layer_conf.input_dims[0][-1]
        self.n_state = self.layer_conf.input_dims[0][-1]
        self.device = torch.device('cuda' if torch.cuda.is_available() else
            'cpu')
        assert self.n_state % self.layer_conf.n_head == 0
        self.c_attn = nn.Linear(self.layer_conf.input_dims[0][-1], self.
            n_state * 3)
        self.c_proj = nn.Linear(self.layer_conf.input_dims[0][-1], self.n_state
            )

    def _attn(self, q, k, v):
        w = torch.matmul(q, k).to(self.device)
        if self.layer_conf.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1000000000.0 * (1 - self.b)
        w = nn.Softmax(dim=-1)(w)
        w = nn.Dropout(self.layer_conf.attn_dropout)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.layer_conf.n_head, x.size(-1) //
            self.layer_conf.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, string, string_len):
        """ process input

        Args:
            string, string_len
            e.g. string (Tensor): [batch_size, seq_len, dim], string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, output_dim], [batch_size]
        """
        self.register_buffer('b', torch.tril(torch.ones(string.shape[1],
            string.shape[1]).to(self.device)).view(1, 1, string.shape[1],
            string.shape[1]))
        x = self.c_attn(string)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = nn.Dropout(self.layer_conf.resid_dropout)(a)
        return a, string_len


class CRFLoss(nn.Module):
    """CRFLoss
       use for crf output layer for sequence tagging task.
    """

    def __init__(self):
        super(CRFLoss, self).__init__()

    def _score_sentence(self, scores, mask, tags, transitions, crf_layer_conf):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if crf_layer_conf.use_gpu:
            new_tags = new_tags
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, (0)] = (tag_size - 2) * tag_size + tags[:, (0)]
            else:
                new_tags[:, (idx)] = tags[:, (idx - 1)] * tag_size + tags[:,
                    (idx)]
        end_transition = transitions[:, (crf_layer_conf.target_dict[
            crf_layer_conf.STOP_TAG])].contiguous().view(1, tag_size).expand(
            batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len,
            batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2,
            new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def forward(self, forward_score, scores, masks, tags, transitions,
        crf_layer_conf):
        """
        
        :param forward_score: Tensor scale
        :param scores: Tensor [seq_len, batch_size, target_size, target_size]
        :param masks:  Tensor [batch_size, seq_len]
        :param tags:   Tensor [batch_size, seq_len]
        :return: goal_score - forward_score
        """
        gold_score = self._score_sentence(scores, masks, tags, transitions,
            crf_layer_conf)
        return forward_score - gold_score


class FocalLoss(nn.Module):
    """ Focal loss
        reference: Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[J]. arXiv preprint arXiv:1708.02002, 2017.
    Args:
        gamma (float): gamma >= 0.
        alpha (float): 0 <= alpha <= 1
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch. However, if the field size_average is set to False, the losses are instead summed for each minibatch. Default is True

    """

    def __init__(self, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = 0
        self.alpha = 0.5
        self.size_average = True
        for key in kwargs:
            setattr(self, key, kwargs[key])
        assert self.alpha <= 1 and self.alpha >= 0, 'The parameter alpha in Focal Loss must be in range [0, 1].'
        if self.alpha is not None:
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])

    def forward(self, input, target):
        """ Get focal loss

        Args:
            input (Variable):  the prediction with shape [batch_size, number of classes]
            target (Variable): the answer with shape [batch_size, number of classes]

        Returns:
            Variable (float): loss
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_microsoft_NeuronBlocks(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BaseLayer(*[], **{'layer_conf': 1}), [], {})

    def test_001(self):
        self._check(Flatten(*[], **{'layer_conf': 1}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ForgetMult(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(QRNN(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(QRNNLayer(*[], **{'input_size': 4}), [torch.rand([4, 4, 4])], {})

