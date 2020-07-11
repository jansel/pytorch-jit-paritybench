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


import time


import numpy as np


import random


import logging


import copy


from queue import Queue


import string


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


from torch.optim import *


import warnings


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


class CNNCharEmbedding(BaseLayer):
    """
    This layer implements the character embedding use CNN
    Args:
        layer_conf (CNNCharEmbeddingConf): configuration of CNNCharEmbedding
    """

    def __init__(self, layer_conf):
        super(CNNCharEmbedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        assert len(layer_conf.dim) == len(layer_conf.window_size) == len(layer_conf.stride), 'The attribute dim/window_size/stride must have the same length.'
        self.char_embeddings = nn.Embedding(layer_conf.vocab_size, layer_conf.embedding_matrix_dim, padding_idx=self.layer_conf.padding)
        nn.init.uniform_(self.char_embeddings.weight, -0.001, 0.001)
        self.char_cnn = nn.ModuleList()
        for i in range(len(layer_conf.output_channel_num)):
            self.char_cnn.append(nn.Conv2d(1, layer_conf.output_channel_num[i], (layer_conf.window_size[i], layer_conf.embedding_matrix_dim), stride=self.layer_conf.stride[i], padding=self.layer_conf.padding))
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None

    def forward(self, string):
        """
        Step1: [batch_size, seq_len, char num in words] -> [batch_size, seq_len * char num in words]
        Step2: lookup embedding matrix -> [batch_size, seq_len * char num in words, embedding_dim]
        reshape -> [batch_size * seq_len, char num in words, embedding_dim]
        Step3: after convolution operation, got [batch_size * seq_len, char num related, output_channel_num]
        Step4: max pooling on axis 1 and -reshape-> [batch_size * seq_len, output_channel_dim]
        Step5: reshape -> [batch_size, seq_len, output_channel_dim]

        Args:
            string (Variable): [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, char num in words]

        Returns:
            Variable: [batch_size, seq_len, output_dim]

        """
        string_reshaped = string.view(string.size()[0], -1)
        char_embs_lookup = self.char_embeddings(string_reshaped).float()
        char_embs_lookup = char_embs_lookup.view(-1, string.size()[2], self.layer_conf.embedding_matrix_dim)
        string_input = torch.unsqueeze(char_embs_lookup, 1)
        outputs = []
        for index, single_cnn in enumerate(self.char_cnn):
            string_conv = single_cnn(string_input).squeeze(3)
            if self.activation:
                string_conv = self.activation(string_conv)
            string_maxpooling = F.max_pool1d(string_conv, string_conv.size(2)).squeeze()
            string_out = string_maxpooling.view(string.size()[0], -1, self.layer_conf.output_channel_num[index])
            outputs.append(string_out)
        if len(outputs) > 1:
            string_output = torch.cat(outputs, 2)
        else:
            string_output = outputs[0]
        return string_output


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


class ConfigurationError(BaseError):
    """ Errors occur when model configuration

    """
    pass


EMBED_LAYER_ID = 'embedding'


class Embedding(BaseLayer):
    """ Embedding layer

    Args:
        layer_conf (EmbeddingConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Embedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.embeddings = nn.ModuleDict() if layer_conf.weight_on_gpu else dict()
        self.char_embeddings = nn.ModuleDict()
        for input_cluster in layer_conf.conf:
            if 'type' in layer_conf.conf[input_cluster]:
                char_emb_conf_dict = copy.deepcopy(layer_conf.conf[input_cluster])
                char_emb_conf_dict['use_gpu'] = layer_conf.use_gpu
                char_emb_conf = eval(layer_conf.conf[input_cluster]['type'] + 'Conf')(**char_emb_conf_dict)
                char_emb_conf.inference()
                char_emb_conf.verify()
                self.char_embeddings[input_cluster] = eval(layer_conf.conf[input_cluster]['type'])(char_emb_conf)
            else:
                self.embeddings[input_cluster] = nn.Embedding(layer_conf.conf[input_cluster]['vocab_size'], layer_conf.conf[input_cluster]['dim'], padding_idx=0)
                if 'init_weights' in layer_conf.conf[input_cluster] and layer_conf.conf[input_cluster]['init_weights'] is not None:
                    self.embeddings[input_cluster].weight = nn.Parameter(torch.from_numpy(layer_conf.conf[input_cluster]['init_weights']))
                if layer_conf.conf[input_cluster]['fix_weight']:
                    self.embeddings[input_cluster].weight.requires_grad = False
                    logging.info("The Embedding[%s][fix_weight] is true, fix the embeddings[%s]'s weight" % (input_cluster, input_cluster))

    def forward(self, inputs, use_gpu=False):
        """ process inputs

        Args:
            inputs (dict): a dictionary to describe each transformer_model inputs. e.g.:

                        char_emb': [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, word character num]

                        'word': word ids (Variable), shape:[batch_size, seq_len],

                        'postag': postag ids (Variable), shape: [batch_size, seq_len],

                        ...
            use_gpu (bool): put embedding matrix on GPU (True) or not (False)

        Returns:
            Variable: the embedding representation with shape [batch_size, seq_len, emb_dim]

        """
        features = []
        for input_cluster in inputs:
            if 'extra' in input_cluster:
                continue
            input = inputs[input_cluster]
            if input_cluster == 'char':
                emb = self.char_embeddings[input_cluster](input).float()
            elif list(self.embeddings[input_cluster].parameters())[0].device.type == 'cpu':
                emb = self.embeddings[input_cluster](input.cpu()).float()
            else:
                emb = self.embeddings[input_cluster](input).float()
            if use_gpu is True:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                emb = emb
            features.append(emb)
        if len(features) > 1:
            return torch.cat(features, 2)
        else:
            return features[0]

    def get_parameters(self):
        for sub_emb in self.embeddings:
            for param in self.embeddings[sub_emb].parameters():
                yield param


class EncoderDecoder(BaseLayer):
    """ The encoder decoder framework

    Args:
        layer_conf (EncoderDecoderConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(EncoderDecoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.encoder = eval(layer_conf.encoder_name)(layer_conf.encoder_conf_cls)
        self.decoder = eval(layer_conf.decoder_name)(layer_conf.decoder_conf_cls)

    def forward(self, string, string_len):
        """ process inputs with encoder & decoder

        Args:
            string (Variable): [batch_size, seq_len, dim]
            string_len (ndarray): [batch_size]

        Returns:
            Variable : decode scores with shape [batch_size, seq_len, decoder_vocab_size]
        """
        encoder_output, encoder_context = self.encoder(string, string_len)
        decoder_scores = self.decoder(string, string_len, encoder_context, encoder_output)
        return decoder_scores, string_len


EMBED_LAYER_NAME = 'Embedding'


class LayerConfigUndefinedError(BaseError):
    """ Errors occur when the corresponding configuration class of a layer is not defined

    """
    pass


def get_conf(layer_id, layer_name, input_layer_ids, all_layer_configs, model_input_ids, use_gpu, conf_dict=None, shared_conf=None, succeed_embedding_flag=False, output_layer_flag=False, target_num=None, fixed_lengths=None, target_dict=None):
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
                        logging.info('#target# position will be replace by target num: %d' % target_num)
                        conf_dict['hidden_dim'][-1] = target_num
                elif isinstance(conf_dict['hidden_dim'], int) and conf_dict['hidden_dim'] == -1:
                    assert output_layer_flag is True, 'Only in the last layer, hidden_dim == -1 is allowed!'
                    assert target_num is not None, 'Number of targets should be given!'
                    conf_dict['hidden_dim'] = target_num
                elif isinstance(conf_dict['hidden_dim'], str) and conf_dict['hidden_dim'] == '#target#':
                    logging.info('#target# position will be replace by target num: %d' % target_num)
                    conf_dict['hidden_dim'] = target_num
            if layer_name == 'CRF':
                conf_dict['target_dict'] = target_dict
            conf = eval(layer_name + 'Conf')(**conf_dict)
        except NameError as e:
            raise LayerConfigUndefinedError('"%sConf" has not been defined' % layer_name)
    if layer_name == EMBED_LAYER_NAME:
        pass
    else:
        for input_layer_id in input_layer_ids:
            if not (input_layer_id in all_layer_configs or input_layer_id in model_input_ids):
                raise ConfigurationError('The input %s of layer %s does not exist. Please define it before defining layer %s!' % (input_layer_id, layer_id, layer_id))
        former_output_ranks = [(all_layer_configs[input_layer_id].output_rank if input_layer_id in all_layer_configs else all_layer_configs[EMBED_LAYER_ID].output_rank) for input_layer_id in input_layer_ids]
        conf.input_dims = [(all_layer_configs[input_layer_id].output_dim if input_layer_id in all_layer_configs else all_layer_configs[EMBED_LAYER_ID].output_dim) for input_layer_id in input_layer_ids]
        if len(input_layer_ids) == 1 and input_layer_ids[0] in model_input_ids and fixed_lengths:
            conf.input_dims[0][1] = fixed_lengths[input_layer_ids[0]]
        if conf.num_of_inputs > 0:
            if conf.num_of_inputs != len(input_layer_ids):
                raise ConfigurationError('%s only accept %d inputs but you feed %d inputs to it!' % (layer_name, conf.num_of_inputs, len(input_layer_ids)))
        elif conf.num_of_inputs == -1:
            conf.num_of_inputs = len(input_layer_ids)
            if isinstance(conf.input_ranks, list):
                conf.input_ranks = conf.input_ranks * conf.num_of_inputs
            else:
                logging.warning('[For developer of %s] The input_ranks attribute should be a list!' % layer_name)
                [conf.input_ranks] * conf.num_of_inputs
        for input_rank, former_output_rank in zip(conf.input_ranks, former_output_ranks):
            if input_rank != -1 and input_rank != former_output_rank:
                raise ConfigurationError('Input ranks of %s are inconsistent with former layers' % layer_id)
        conf.input_ranks = copy.deepcopy(former_output_ranks)
    conf.inference()
    conf.verify()
    former_conf = None if len(all_layer_configs) == 0 else list(all_layer_configs.values())[-1]
    conf.verify_former_block(former_conf)
    logging.debug('Layer id: %s; name: %s; input_dims: %s; input_ranks: %s; output_dim: %s; output_rank: %s' % (layer_id, layer_name, conf.input_dims if layer_id != 'embedding' else 'None', conf.input_ranks, conf.output_dim, conf.output_rank))
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
        raise Exception('%s; Layer "%s" has not been defined' % (str(e), layer_name))
    return layer


def transfer_to_gpu(cpu_element):
    """

    Args:
        cpu_element: either a tensor or a module

    Returns:

    """
    return cpu_element


def transform_tensors2params(inputs_desc, length_desc, param_list):
    """ Inverse function of transform_params2tensors

    Args:
        param_list:
        inputs_desc:
        length_desc:

    Returns:

    """
    inputs = {}
    for key in inputs_desc:
        input, input_type = key.split('___')
        if not input in inputs:
            inputs[input] = dict()
        inputs[input][input_type] = param_list[inputs_desc[key]]
    lengths = {}
    for key in length_desc:
        if '__' in key:
            input, input_type = key.split('__')
            if not input in lengths:
                lengths[input] = dict()
            lengths[input][input_type] = param_list[length_desc[key]]
        else:
            lengths[key] = param_list[length_desc[key]]
    return inputs, lengths


class Model(nn.Module):

    def __init__(self, conf, problem, vocab_info, use_gpu):
        """

        Args:
            inputs: ['string1', 'string2']
            layer_archs:  The layers must produce tensors with similar shapes. The layers may be nested.
                [
                    {
                    'layer': Layer name,
                    'conf': {xxxx}
                    },
                    [
                        {
                        'layer': Layer name,
                        'conf': {},
                        },
                        {
                        'layer': Layer name,
                        'conf': {},
                        }
                    ]
                ]
            vocab_info:
                {
                    'word':  {
                        'vocab_size': xxx,
                        'init_weights': np matrix
                        }
                    'postag': {
                        'vocab_size': xxx,
                        'init_weights': None
                        }
                }
        """
        super(Model, self).__init__()
        inputs = conf.object_inputs_names
        layer_archs = conf.architecture
        target_num = problem.output_target_num()
        if conf.fixed_lengths:
            fixed_lengths_corrected = copy.deepcopy(conf.fixed_lengths)
            for seq in fixed_lengths_corrected:
                if problem.with_bos_eos:
                    fixed_lengths_corrected[seq] += 2
        else:
            fixed_lengths_corrected = None
        self.use_gpu = use_gpu
        all_layer_configs = dict()
        self.layers = nn.ModuleDict()
        self.layer_inputs = dict()
        self.layer_dependencies = dict()
        self.layer_dependencies[EMBED_LAYER_ID] = set()
        self.output_layer_id = []
        for layer_index, layer_arch in enumerate(layer_archs):
            output_layer_flag = True if 'output_layer_flag' in layer_arch and layer_arch['output_layer_flag'] is True else False
            succeed_embedding_flag = True if layer_index > 0 and 'inputs' in layer_arch and [(input in inputs) for input in layer_arch['inputs']].count(True) == len(layer_arch['inputs']) else False
            if output_layer_flag:
                self.output_layer_id.append(layer_arch['layer_id'])
            if layer_index == 0:
                emb_conf = copy.deepcopy(vocab_info)
                for input_cluster in emb_conf:
                    emb_conf[input_cluster]['dim'] = layer_arch['conf'][input_cluster]['dim']
                    emb_conf[input_cluster]['fix_weight'] = layer_arch['conf'][input_cluster].get('fix_weight', False)
                emb_conf['weight_on_gpu'] = layer_arch.get('weight_on_gpu', True)
                all_layer_configs[EMBED_LAYER_ID] = get_conf(EMBED_LAYER_ID, layer_arch['layer'], None, all_layer_configs, inputs, self.use_gpu, conf_dict={'conf': emb_conf}, shared_conf=None, succeed_embedding_flag=False, output_layer_flag=output_layer_flag, target_num=target_num, fixed_lengths=fixed_lengths_corrected, target_dict=problem.output_dict)
                self.add_layer(EMBED_LAYER_ID, get_layer(layer_arch['layer'], all_layer_configs[EMBED_LAYER_ID]))
            else:
                if layer_arch['layer'] in self.layers and not 'conf' in layer_arch:
                    logging.debug('Layer id: %s; Sharing configuration with layer %s' % (layer_arch['layer_id'], layer_arch['layer']))
                    conf_dict = None
                    shared_conf = all_layer_configs[layer_arch['layer']]
                else:
                    conf_dict = layer_arch['conf']
                    shared_conf = None
                if layer_arch['layer'] == 'EncoderDecoder':
                    layer_arch['conf']['decoder_conf']['decoder_vocab_size'] = target_num
                all_layer_configs[layer_arch['layer_id']] = get_conf(layer_arch['layer_id'], layer_arch['layer'], layer_arch['inputs'], all_layer_configs, inputs, self.use_gpu, conf_dict=conf_dict, shared_conf=shared_conf, succeed_embedding_flag=succeed_embedding_flag, output_layer_flag=output_layer_flag, target_num=target_num, fixed_lengths=fixed_lengths_corrected, target_dict=problem.output_dict)
                if layer_arch['layer'] in self.layers and not 'conf' in layer_arch:
                    self.add_layer(layer_arch['layer_id'], self.layers[layer_arch['layer']])
                else:
                    self.add_layer(layer_arch['layer_id'], get_layer(layer_arch['layer'], all_layer_configs[layer_arch['layer_id']]))
                self.layer_inputs[layer_arch['layer_id']] = layer_arch['inputs']
                cur_layer_depend = set()
                for layer_depend_id in layer_arch['inputs']:
                    if not layer_depend_id in inputs:
                        cur_layer_depend.add(layer_depend_id)
                self.add_dependency(layer_arch['layer_id'], cur_layer_depend)
        logging.debug('Layer dependencies: %s' % repr(self.layer_dependencies))
        if not hasattr(self, 'output_layer_id'):
            raise ConfigurationError('Please define an output layer')
        self.layer_topological_sequence = self.get_topological_sequence()

    def add_layer(self, layer_id, layer):
        """ register a layer

        Args:
            layer_id:
            layer:

        Returns:

        """
        if layer_id in self.layers:
            raise ConfigurationError('The layer id %s is not unique!')
        else:
            self.layers[layer_id] = layer

    def add_dependency(self, layer_id, depend_layer_id):
        """ add the layers have to be proceed before layer_id

        Args:
            layer_id:
            depend_layer_id:

        Returns:

        """
        if not layer_id in self.layer_dependencies:
            self.layer_dependencies[layer_id] = set()
        if isinstance(depend_layer_id, int):
            self.layer_dependencies[layer_id].add(depend_layer_id)
        else:
            self.layer_dependencies[layer_id] |= set(depend_layer_id)

    def remove_dependency(self, depend_layer_id):
        """ remove dependencies on layer_id

        Args:
            layer_id:

        Returns:

        """
        for layer_id in self.layer_dependencies:
            self.layer_dependencies[layer_id].remove(depend_layer_id)

    def get_topological_sequence(self):
        """ get topological sequence of nodes in the model

        Returns:

        """
        total_layer_ids = Queue()
        for layer_id in self.layers.keys():
            if layer_id != EMBED_LAYER_ID:
                total_layer_ids.put(layer_id)
        topological_list = []
        circular_cnt = 0
        while not total_layer_ids.empty():
            layer_id = total_layer_ids.get()
            if len(self.layer_dependencies[layer_id]) == 0:
                for layer_id2 in self.layer_dependencies:
                    if layer_id in self.layer_dependencies[layer_id2]:
                        self.layer_dependencies[layer_id2].remove(layer_id)
                circular_cnt = 0
                topological_list.append(layer_id)
            else:
                total_layer_ids.put(layer_id)
                circular_cnt += 1
                if circular_cnt >= total_layer_ids.qsize():
                    rest_layers = []
                    while not total_layer_ids.empty():
                        rest_layers.append(total_layer_ids.get())
                    raise ConfigurationError('The model architecture is illegal because there is a circular dependency or there are some isolated layers. The layers can not be resolved: [%s]' % ', '.join(rest_layers))
        logging.debug('Topological sequence of nodes: %s' % ','.join(topological_list))
        return topological_list

    def forward(self, inputs_desc, length_desc, *param_list):
        """

        Args:
            with the help of transform_tensors2params(inputs_desc, length_desc, param_list), we can get the below inputs and lengths

            inputs: dict.
                {
                    "string1":{
                        'word': word ids, [batch size, seq len]
                        'postag': postag ids,[batch size, seq len]
                        ...
                    }
                    "string2":{
                        'word': word ids,[batch size, seq len]
                        'postag': postag ids,[batch size, seq len]
                        ...
                    }
                }
            lengths: dict.
                {
                    "string1": [...]
                    "string2": [...]
                }

        Returns:

        """
        inputs, lengths = transform_tensors2params(inputs_desc, length_desc, param_list)
        representation = dict()
        representation[EMBED_LAYER_ID] = dict()
        repre_lengths = dict()
        repre_lengths[EMBED_LAYER_ID] = dict()
        for input in inputs:
            representation[input] = self.layers[EMBED_LAYER_ID](inputs[input], use_gpu=self.is_cuda())
            if self.use_gpu:
                repre_lengths[input] = transfer_to_gpu(lengths[input])
            else:
                repre_lengths[input] = lengths[input]
        for layer_id in self.layer_topological_sequence:
            input_params = []
            for input_layer_id in self.layer_inputs[layer_id]:
                input_params.append(representation[input_layer_id])
                input_params.append(repre_lengths[input_layer_id])
            representation[layer_id], repre_lengths[layer_id] = self.layers[layer_id](*input_params)
        representation_output = dict()
        for single_output_layer_id in self.output_layer_id:
            representation_output[single_output_layer_id] = representation[single_output_layer_id]
        return representation_output

    def is_cuda(self):
        return list(self.parameters())[-1].data.is_cuda

    def update_use_gpu(self, new_use_gpu):
        self.use_gpu = new_use_gpu
        for layer_id in self.layers.keys():
            if isinstance(self.layers[layer_id], Embedding):
                for input_cluster in self.layers[layer_id].embeddings:
                    if isinstance(self.layers[layer_id].embeddings[input_cluster], CNNCharEmbedding):
                        self.layers[layer_id].embeddings[input_cluster].layer_conf.use_gpu = new_use_gpu
            elif isinstance(self.layers[layer_id], EncoderDecoder):
                self.layers[layer_id].encoder.layer_conf.use_gpu = new_use_gpu
                self.layers[layer_id].decoder.layer_conf.use_gpu = new_use_gpu
            else:
                self.layers[layer_id].layer_conf.use_gpu = new_use_gpu


class BiGRU(BaseLayer):
    """Bidirectional GRU

    Args:
        layer_conf (BiGRUConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiGRU, self).__init__(layer_conf)
        self.GRU = nn.GRU(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, 1, bidirectional=True, dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        padded_seq_len = string.shape[1]
        self.init_GRU = torch.FloatTensor(2, string.size(0), self.layer_conf.hidden_dim).zero_()
        if self.is_cuda():
            self.init_GRU = transfer_to_gpu(self.init_GRU)
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]
        string = string.index_select(0, idx_sort)
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.GRU.flatten_parameters()
        string_output, hn = self.GRU(string_packed, self.init_GRU)
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]
        string_output = string_output.index_select(0, idx_unsort)
        return string_output, string_len


class BiGRULast(BaseLayer):
    """ Get the last hidden state of Bi GRU

    Args:
        layer_conf (BiGRULastConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiGRULast, self).__init__(layer_conf)
        self.GRU = nn.GRU(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, 1, bidirectional=True, dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, 2 * hidden_dim]
        """
        self.init_GRU = torch.FloatTensor(2, string.size(0), self.layer_conf.hidden_dim).zero_()
        if self.is_cuda():
            self.init_GRU = transfer_to_gpu(self.init_GRU)
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]
        string = string.index_select(0, idx_sort)
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.GRU.flatten_parameters()
        string_output, hn = self.GRU(string_packed, self.init_GRU)
        emb = torch.cat((hn[0], hn[1]), 1)
        emb = emb.index_select(0, idx_unsort)
        return emb, string_len


class BiLSTM(BaseLayer):
    """ Bidrectional LSTM

    Args:
        layer_conf (BiLSTMConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiLSTM, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True, dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        padded_seq_len = string.shape[1]
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]
        string = string.index_select(0, idx_sort)
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.lstm.flatten_parameters()
        string_output = self.lstm(string_packed)[0]
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]
        string_output = string_output.index_select(0, idx_unsort)
        return string_output, string_len


class BiLSTMAtt(BaseLayer):
    """ BiLSTM with self attention

    Args:
        layer_conf (BiLSTMAttConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiLSTMAtt, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True, dropout=layer_conf.dropout, batch_first=True)
        self.att = nn.Parameter(torch.randn(layer_conf.attention_dim, layer_conf.attention_dim), requires_grad=True)
        nn.init.uniform_(self.att, a=0, b=1)
        self.softmax = nn.Softmax()

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        padded_seq_len = string.shape[1]
        string_len_sorted, idx_sort = (-string_len).sort()
        string_len_sorted = -string_len_sorted
        idx_unsort = idx_sort.sort()[1]
        bsize = string.shape[0]
        string = string.index_select(0, idx_sort)
        string_packed = nn.utils.rnn.pack_padded_sequence(string, string_len_sorted, batch_first=True)
        self.lstm.flatten_parameters()
        string_output = self.lstm(string_packed)[0]
        string_output = nn.utils.rnn.pad_packed_sequence(string_output, batch_first=True, total_length=padded_seq_len)[0]
        string_output = string_output.index_select(0, idx_unsort).contiguous()
        alphas = string_output.matmul(self.att).bmm(string_output.transpose(1, 2).contiguous())
        alphas = alphas + (alphas == 0).float() * -10000
        alphas = self.softmax(alphas.view(-1, int(padded_seq_len)))
        alphas = alphas.view(bsize, -1, int(padded_seq_len))
        string_output = alphas.bmm(string_output)
        return string_output, string_len


class BiLSTMLast(BaseLayer):
    """ get last hidden states of Bidrectional LSTM

    Args:
        layer_conf (BiLSTMConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiLSTMLast, self).__init__(layer_conf)
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, bidirectional=True, dropout=layer_conf.dropout, batch_first=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, 2 * hidden_dim]

        """
        str_len, idx_sort = (-string_len).sort()
        str_len = -str_len
        idx_unsort = idx_sort.sort()[1]
        string = string.index_select(0, idx_sort)
        string_packed = nn.utils.rnn.pack_padded_sequence(string, str_len, batch_first=True)
        self.lstm.flatten_parameters()
        string_output, (hn, cn) = self.lstm(string_packed)
        emb = torch.cat((hn[0], hn[1]), 1)
        emb = emb.index_select(0, idx_unsort)
        return emb, string_len


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

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()
        assert window in [1, 2], 'This QRNN implementation currently only handles convolutional window of size 1 or size 2'
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()
        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
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
                mask = F.new_empty(F.size(), requires_grad=False).bernoulli_(1 - self.zoneout)
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

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, **kwargs):
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'
        super(QRNN, self).__init__()
        if bidirectional:
            self.layers = torch.nn.ModuleList([QRNNLayer(input_size if l < 2 else hidden_size * 2, hidden_size, **kwargs) for l in range(num_layers * 2)])
        else:
            self.layers = torch.nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
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
                output, hn = layer(input, None if hidden is None else hidden[l])
                next_hidden.append(hn)
                if j == 1:
                    output = self.tensor_reverse(output)
                all_output.append(output)
            input = torch.cat(all_output, input.dim() - 1)
            if self.dropout != 0 and i < self.num_layers - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)
        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers * self.num_directions, *next_hidden[0].size()[-2:])
        return input, next_hidden


class BiQRNN(BaseLayer):
    """ Bidrectional QRNN

    Args:
        layer_conf (BiQRNNConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(BiQRNN, self).__init__(layer_conf)
        self.qrnn = QRNN(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, window=layer_conf.window, zoneout=layer_conf.zoneout, dropout=layer_conf.dropout, bidirectional=True)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        string = string.transpose(0, 1)
        string_output = self.qrnn(string)[0]
        string_output = string_output.transpose(0, 1)
        return string_output, string_len


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(BaseLayer):
    """ Conditional Random Field layer

    Args:
        layer_conf(CRFConf): configuration of CRF layer
    """

    def __init__(self, layer_conf):
        super(CRF, self).__init__(layer_conf)
        self.target_size = len(self.layer_conf.target_dict)
        init_transitions = torch.zeros(self.target_size, self.target_size)
        init_transitions[:, (self.layer_conf.target_dict[self.layer_conf.START_TAG])] = -10000.0
        init_transitions[(self.layer_conf.target_dict[self.layer_conf.STOP_TAG]), :] = -10000.0
        init_transitions[:, (0)] = -10000.0
        init_transitions[(0), :] = -10000.0
        if self.layer_conf.use_gpu:
            init_transitions = init_transitions
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_forward(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        partition = inivalues[:, (self.layer_conf.target_dict[self.layer_conf.START_TAG]), :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            mask_idx = mask[(idx), :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, (self.layer_conf.target_dict[self.layer_conf.STOP_TAG])]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, (self.layer_conf.target_dict[self.layer_conf.START_TAG]), :].clone().view(batch_size, tag_size)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.layer_conf.use_gpu:
            pad_zero = pad_zero
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        pointer = last_bp[:, (self.layer_conf.target_dict[self.layer_conf.STOP_TAG])]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.layer_conf.use_gpu:
            decode_idx = decode_idx
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, string, string_len):
        """
        CRF layer process: include use transition matrix compute score and  viterbi decode

        Args:
            string(Tensor): [batch_size, seq_len, target_num]
            string_len(Tensor): [batch_size]

        Returns:
            score: the score by CRF inference
            best_path: the best bath of viterbi decode
        """
        assert string_len is not None, 'CRF layer need string length for mask.'
        masks = []
        string_len_val = string_len.cpu().data.numpy()
        for i in range(len(string_len)):
            masks.append(torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
        masks = torch.stack(masks).view(string.shape[0], string.shape[1]).byte()
        if self.layer_conf.use_gpu:
            masks = masks
        forward_score, scores = self._calculate_forward(string, masks)
        _, tag_seq = self._viterbi_decode(string, masks)
        return (forward_score, scores, masks, tag_seq, self.transitions, self.layer_conf), string_len


class Conv(BaseLayer):
    """ Convolution along just 1 direction

    Args:
        layer_conf (ConvConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Conv, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None
        self.conv = nn.Conv1d(layer_conf.input_dims[0][-1], layer_conf.output_channel_num, kernel_size=layer_conf.window_size, padding=layer_conf.padding)
        if layer_conf.batch_norm:
            self.batch_norm = nn.BatchNorm1d(layer_conf.output_channel_num)
        else:
            self.batch_norm = None
        if layer_conf.dropout > 0:
            self.cov_dropout = nn.Dropout(layer_conf.dropout)
        else:
            self.cov_dropout = None
        if layer_conf.use_gpu:
            self.conv = self.conv
            if self.batch_norm:
                self.batch_norm = self.batch_norm
            if self.cov_dropout:
                self.cov_dropout = self.cov_dropout
            if self.activation:
                self.activation = self.activation

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, seq_len, feature_dim]
            string_len (Tensor):  [batch_size]

        Returns:
            Tensor: shape: [batch_size, (seq_len - conv_window_size) // stride + 1, output_channel_num]

        """
        if string_len is not None and self.layer_conf.mask:
            string_len_val = string_len.cpu().data.numpy()
            masks = []
            for i in range(len(string_len)):
                masks.append(torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
            masks = torch.stack(masks).view(string.shape[0], string.shape[1], 1).expand_as(string)
            if self.is_cuda():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                masks = masks
            string = string * masks
        string_ = string.transpose(2, 1).contiguous()
        string_out = self.conv(string_)
        if self.activation:
            string_out = self.activation(string_out)
        if self.cov_dropout:
            string_out = self.cov_dropout(string_out)
        if self.batch_norm:
            string_out = self.batch_norm(string_out)
        string_out = string_out.transpose(2, 1).contiguous()
        string_len_out = None
        if string_len is not None and self.layer_conf.remind_lengths:
            string_len_out = string_len
        return string_out, string_len_out


class Conv2D(BaseLayer):
    """ Convolution along just 1 direction

    Args:
        layer_conf (ConvConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Conv2D, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None
        self.cnn = nn.Conv2d(in_channels=layer_conf.input_channel_num, out_channels=layer_conf.output_channel_num, kernel_size=layer_conf.window_size, stride=layer_conf.stride, padding=layer_conf.padding)
        if layer_conf.batch_norm:
            self.batch_norm = nn.BatchNorm2d(layer_conf.output_channel_num)
        else:
            self.batch_norm = None

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, width, feature_dim]
            string_len (Tensor):  [batch_size]

        Returns:
            Tensor: shape: [batch_size, (length - conv_window_size) // stride + 1, (width - conv_window_size) // stride + 1, output_channel_num]

        """
        string = string.permute([0, 3, 1, 2]).contiguous()
        string_out = self.cnn(string)
        if hasattr(self, 'batch_norms') and self.batch_norm:
            string_out = self.batch_norm(string_out)
        string_out = string_out.permute([0, 2, 3, 1]).contiguous()
        if self.activation:
            string_out = self.activation(string_out)
        if string_len is not None:
            string_len_out = (string_len - self.layer_conf.window_size[0]) // self.layer_conf.stride[0] + 1
        else:
            string_len_out = None
        return string_out, string_len_out


class ConvPooling(BaseLayer):
    """ Convolution along just 1 direction

    Args:
        layer_conf (ConvConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(ConvPooling, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.filters = nn.ParameterList()
        if layer_conf.batch_norm:
            self.batch_norms = nn.ModuleList()
        else:
            self.batch_norms = None
        for i in range(len(layer_conf.window_sizes)):
            self.filters.append(nn.Parameter(torch.randn(layer_conf.output_channel_num, layer_conf.input_channel_num, layer_conf.window_sizes[i], layer_conf.input_dims[0][2], requires_grad=True).float()))
            if layer_conf.batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(layer_conf.output_channel_num))
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, seq_len, feature_dim]
            string_len (Tensor):  [batch_size]

        Returns:
            Tensor: shape: [batch_size, (seq_len - conv_window_size) // stride + 1, output_channel_num]

        """
        if string_len is not None:
            string_len_val = string_len.cpu().data.numpy()
            masks = []
            for i in range(len(string_len)):
                masks.append(torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
            masks = torch.stack(masks).view(string.shape[0], string.shape[1], 1).expand_as(string)
            if self.is_cuda():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                masks = masks
            string = string * masks
        string = torch.unsqueeze(string, 1)
        outputs = []
        for idx, (filter, window_size) in enumerate(zip(self.filters, self.layer_conf.window_sizes)):
            string_out = F.conv2d(string, filter, stride=self.layer_conf.stride, padding=self.layer_conf.padding)
            if hasattr(self, 'batch_norms') and self.batch_norms:
                string_out = self.batch_norms[idx](string_out)
            string_out = torch.squeeze(string_out, 3).permute(0, 2, 1)
            if self.activation:
                string_out = self.activation(string_out)
            if string_len is not None:
                string_len_out = (string_len - window_size) // self.layer_conf.stride + 1
            else:
                string_len_out = None
            if self.layer_conf.pool_type == 'mean':
                assert not string_len_out is None, 'Parameter string_len should not be None!'
                string_out = torch.sum(string_out, self.layer_conf.pool_axis).squeeze(self.layer_conf.pool_axis)
                if not torch.is_tensor(string_len_out):
                    string_len_out = torch.FloatTensor(string_len_out)
                string_len_out = string_len_out.unsqueeze(1)
                if self.is_cuda():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    string_len_out = string_len_out
                output = string_out / string_len_out.expand_as(string_out)
            elif self.layer_conf.pool_type == 'max':
                output = torch.max(string_out, self.layer_conf.pool_axis)[0]
            outputs.append(output)
        if len(outputs) > 1:
            string_output = torch.cat(outputs, 1)
        else:
            string_output = outputs[0]
        return string_output, None


class Dropout(BaseLayer):
    """ Dropout

    Args:
        layer_conf (DropoutConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Dropout, self).__init__(layer_conf)
        self.dropout_layer = nn.Dropout(layer_conf.dropout)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: has the same shape as string.
        """
        string_out = self.dropout_layer(string)
        return string_out, string_len


class HighwayLinear(BaseLayer):
    """ A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Args:
        layer_conf (HighwayLinearConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(HighwayLinear, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1] * 2) for _ in range(layer_conf.num_layers)])
        self.activation = eval('nn.' + layer_conf.activation)()

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Tensor): [batch_size, seq_len, dim]
            string_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len, 2 * hidden_dim]

        """
        current_input = string
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input, string_len


class Linear(BaseLayer):
    """ Linear layer

    Args:
        layer_conf (LinearConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Linear, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        if layer_conf.input_ranks[0] == 3 and layer_conf.batch_norm is True:
            layer_conf.batch_norm = False
            logging.warning('Batch normalization for dense layers of which the rank is 3 is not available now. Batch norm is set to False now.')
        if isinstance(layer_conf.hidden_dim, int):
            layer_conf.hidden_dim = [layer_conf.hidden_dim]
        layers = OrderedDict()
        former_dim = layer_conf.input_dims[0][-1]
        for i in range(len(layer_conf.hidden_dim)):
            layers['linear_%d' % len(layers)] = nn.Linear(former_dim, layer_conf.hidden_dim[i])
            if layer_conf.activation is not None and (layer_conf.last_hidden_activation is True or i != len(layer_conf.hidden_dim) - 1):
                try:
                    if layer_conf.batch_norm:
                        layers['batch_norm_%d' % len(layers)] = nn.BatchNorm1d(layer_conf.hidden_dim[i])
                    layers['linear_activate_%d' % len(layers)] = eval('nn.' + layer_conf.activation)()
                except NameError as e:
                    raise Exception('%s; Activation layer "nn.%s"' % (str(e), layer_conf.activation))
            if layer_conf.last_hidden_softmax is True and i == len(layer_conf.hidden_dim) - 1:
                layers['linear_softmax_%d' % len(layers)] = nn.Softmax(layer_conf.output_rank - 1)
            former_dim = layer_conf.hidden_dim[i]
        self.linear = nn.Sequential(layers)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: has the same shape as string.
        """
        if self.layer_conf.input_ranks[0] == 3 and string_len is not None:
            string_len_val = string_len.cpu().data.numpy()
            masks = []
            for i in range(len(string_len)):
                masks.append(torch.cat([torch.ones(string_len_val[i]), torch.zeros(string.shape[1] - string_len_val[i])]))
            masks = torch.stack(masks).view(string.shape[0], string.shape[1], 1).expand_as(string)
            if self.is_cuda():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                masks = masks
            string = string * masks
        string_out = self.linear(string.float())
        if not self.layer_conf.keep_dim:
            string_out = torch.squeeze(string_out, -1)
        return string_out, string_len


class Pooling(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Pooling, self).__init__(layer_conf)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): any shape.
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """
        if self.layer_conf.pool_type == 'mean':
            assert not string_len is None, 'Parameter string_len should not be None!'
            string = torch.sum(string, self.layer_conf.pool_axis).squeeze(self.layer_conf.pool_axis)
            if not torch.is_tensor(string_len):
                string_len = torch.FloatTensor(string_len).unsqueeze(1)
            if self.is_cuda():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                string_len = string_len
            string_len = string_len.unsqueeze(1)
            output = string / string_len.expand_as(string).float()
        elif self.layer_conf.pool_type == 'max':
            output = torch.max(string, self.layer_conf.pool_axis)[0]
        return output, string_len


class Pooling1D(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Pooling1D, self).__init__(layer_conf)
        self.pool = None
        if layer_conf.pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=layer_conf.window_size, stride=layer_conf.stride, padding=layer_conf.padding)
        elif layer_conf.pool_type == 'mean':
            self.pool = nn.AvgPool1d(kernel_size=layer_conf.window_size, stride=layer_conf.stride, padding=layer_conf.padding)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, feature_dim]
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """
        string = string.permute([0, 2, 1]).contiguous()
        string = self.pool(string)
        string = string.permute([0, 2, 1]).contiguous()
        return string, string_len


class Pooling2D(BaseLayer):
    """ Pooling layer

    Args:
        layer_conf (PoolingConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Pooling2D, self).__init__(layer_conf)
        self.pool = None
        if layer_conf.pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=layer_conf.window_size, stride=layer_conf.stride, padding=layer_conf.padding)
        elif layer_conf.pool_type == 'mean':
            self.pool = nn.AvgPool2d(kernel_size=layer_conf.window_size, stride=layer_conf.stride, padding=layer_conf.padding)

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Tensor): tensor with shape: [batch_size, length, width, feature_dim]
            string_len (Tensor): [batch_size], default is None.

        Returns:
            Tensor: Pooling result of string

        """
        string = string.permute([0, 3, 1, 2]).contiguous()
        string = self.pool(string)
        string = string.permute([0, 2, 3, 1]).contiguous()
        return string, string_len


class Transformer(nn.Module):
    """ Transformer layer

    Args:
        layer_conf (TransformerConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Transformer, self).__init__()
        self.layer_conf = layer_conf
        self.transformer_layer = nn.ModuleList([copy.deepcopy(nn.ModuleList([eval(layer_conf.attention_name)(layer_conf.attention_conf_cls), eval(layer_conf.layernorm1_name)(layer_conf.layernorm1_conf_cls), eval(layer_conf.mlp_name)(layer_conf.mlp_conf_cls), eval(layer_conf.layernorm2_name)(layer_conf.layernorm2_conf_cls)])) for _ in range(self.layer_conf.n_layer)])

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


class Attention(BaseLayer):
    """  Attention layer

    Given sequences X and Y, match sequence Y to each element in X.

    Args:
        layer_conf (AttentionConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Attention, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x (Tensor):      [batch_size, x_max_len, dim].
            x_len (Tensor):  [batch_size], default is None.
            y (Tensor):      [batch_size, y_max_len, dim].
            y_len(Tensor):  [batch_size], default is None.

        Returns:
            output: has the same shape as x.

        """
        scores = x.bmm(y.transpose(2, 1))
        batch_size, y_max_len, _ = y.size()
        y_length = y_len.cpu().numpy()
        y_mask = np.ones((batch_size, y_max_len))
        for i, single_len in enumerate(y_length):
            y_mask[i][:single_len] = 0
        y_mask = torch.from_numpy(y_mask).byte()
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, float('-inf'))
        alpha = self.softmax(scores)
        output = alpha.bmm(y)
        return output, x_len


class BiAttFlow(BaseLayer):
    """
    implement AttentionFlow layer for BiDAF
    [paper]: https://arxiv.org/pdf/1611.01603.pdf

    Args:
        layer_conf(AttentionFlowConf): configuration of the AttentionFlowConf
    """

    def __init__(self, layer_conf):
        super(BiAttFlow, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.attention_weight_content = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_weight_query = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_weight_cq = nn.Linear(layer_conf.input_dims[0][-1], 1)
        self.attention_dropout = nn.Dropout(layer_conf.attention_dropout)

    def forward(self, content, content_len, query, query_len=None):
        """
        implement the attention flow layer of BiDAF model

        :param content (Tensor): [batch_size, content_seq_len, dim]
        :param content_len: [batch_size]
        :param query (Tensor): [batch_size, query_seq_len, dim]
        :param query_len: [batch_size]
        :return: the tensor has same shape as content
        """
        assert content.size()[2] == query.size()[2], 'The dimension of axis 2 of content and query must be consistent! But now, content.size() is %s and query.size() is %s' % (content.size(), query.size())
        batch_size = content.size()[0]
        content_seq_len = content.size()[1]
        query_seq_len = query.size()[1]
        feature_dim = content.size()[2]
        content_aug = content.unsqueeze(2).expand(batch_size, content_seq_len, query_seq_len, feature_dim)
        query_aug = query.unsqueeze(1).expand(batch_size, content_seq_len, query_seq_len, feature_dim)
        content_aug = content_aug.contiguous().view(batch_size * content_seq_len * query_seq_len, feature_dim)
        query_aug = query_aug.contiguous().view(batch_size * content_seq_len * query_seq_len, feature_dim)
        content_query_comb = torch.cat((query_aug, content_aug, content_aug * query_aug), 1)
        attention = self.W(content_query_comb)
        attention = self.attention_dropout(attention)
        attention = attention.view(batch_size, content_seq_len, query_seq_len)
        attention_logits = F.softmax(attention, dim=2)
        content2query_att = torch.bmm(attention_logits, query)
        b = F.softmax(torch.max(attention, dim=2)[0], dim=1).unsqueeze(1)
        query2content_att = torch.bmm(b, content).squeeze(1)
        query2content_att = query2content_att.unsqueeze(1).expand(-1, content_seq_len, -1)
        result = torch.cat([content, content2query_att, content * content2query_att, content * query2content_att], dim=-1)
        return result, content_len


class BilinearAttention(BaseLayer):
    """  BilinearAttention layer for DrQA
    [paper]  https://arxiv.org/abs/1704.00051
    Args:
        layer_conf (BilinearAttentionConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(BilinearAttention, self).__init__(layer_conf)
        self.linear = nn.Linear(layer_conf.input_dims[1][-1], layer_conf.input_dims[0][-1])

    def forward(self, x, x_len, y, y_len):
        """ process inputs

        Args:
            x (Tensor):      [batch_size, x_len, x_dim].
            x_len (Tensor):  [batch_size], default is None.
            y (Tensor):      [batch_size, y_dim].
            y_len (Tensor):  [batch_size], default is None.
        Returns:
            output: [batch_size, x_len, 1].
            x_len:

        """
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2))
        return xWy, x_len


class FullAttention(BaseLayer):
    """ Full-aware fusion of:
            Via, U., With, T., & To, P. (2018). Fusion Net: Fusing Via Fully-Aware Attention with Application to Machine Comprehension, 1–17.

    """

    def __init__(self, layer_conf):
        super(FullAttention, self).__init__(layer_conf)
        self.layer_conf.hidden_dim = layer_conf.hidden_dim
        self.linear = nn.Linear(layer_conf.input_dims[2][-1], layer_conf.hidden_dim, bias=False)
        if layer_conf.input_dims[2][-1] == layer_conf.input_dims[3][-1]:
            self.linear2 = self.linear
        else:
            self.linear2 = nn.Linear(layer_conf.input_dims[3][-1], layer_conf.hidden_dim, bias=False)
        self.linear_final = Parameter(torch.ones(1, layer_conf.hidden_dim), requires_grad=True)
        self.activation = eval('nn.' + layer_conf.activation)()

    def forward(self, string1, string1_len, string2, string2_len, string1_HoW, string1_How_len, string2_HoW, string2_HoW_len):
        """ To get representation of string1, we use string1 and string2 to obtain attention weights and use string2 to represent string1

        Note: actually, the semantic information of string1 is not used, we only need string1's seq_len information

        Args:
            string1: [batch size, seq_len, input_dim1]
            string1_len: [batch_size]
            string2: [batch size, seq_len, input_dim2]
            string2_len: [batch_size]
            string1_HoW: [batch size, seq_len, att_dim1]
            string1_HoW_len: [batch_size]
            string2_HoW: [batch size, seq_len, att_dim2]
            string2_HoW_len: [batch_size]

        Returns:
            string1's representation
            string1_len

        """
        string1_key = self.activation(self.linear(string1_HoW.contiguous().view(-1, string1_HoW.size()[2])))
        string2_key = self.activation(self.linear2(string2_HoW.contiguous().view(-1, string2_HoW.size()[2])))
        final_v = self.linear_final.expand_as(string2_key)
        string2_key = final_v * string2_key
        string1_rep = string1_key.view(-1, string1.size(1), 1, self.layer_conf.hidden_dim).transpose(1, 2).contiguous().view(-1, string1.size(1), self.layer_conf.hidden_dim)
        string2_rep = string2_key.view(-1, string2.size(1), 1, self.layer_conf.hidden_dim).transpose(1, 2).contiguous().view(-1, string2.size(1), self.layer_conf.hidden_dim)
        scores = string1_rep.bmm(string2_rep.transpose(1, 2)).view(-1, 1, string1.size(1), string2.size(1))
        string2_len_np = string2_len.cpu().numpy()
        if torch.cuda.device_count() > 1:
            string2_max_len = string2.shape[1]
        else:
            string2_max_len = string2_len_np.max()
        string2_mask = np.array([([0] * num + [1] * (string2_max_len - num)) for num in string2_len_np])
        string2_mask = torch.from_numpy(string2_mask).unsqueeze(1).unsqueeze(2).expand_as(scores)
        if self.is_cuda():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            string2_mask = string2_mask
        scores.data.masked_fill_(string2_mask.data.byte(), -float('inf'))
        alpha_flat = F.softmax(scores.view(-1, string2.size(1)), dim=1)
        alpha = alpha_flat.view(-1, string1.size(1), string2.size(1))
        string1_atten_seq = alpha.bmm(string2)
        return string1_atten_seq, string1_len


class Interaction(BaseLayer):
    """Bidirectional GRU

    Args:
        layer_conf (BiGRUConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Interaction, self).__init__(layer_conf)
        self.matching_type = layer_conf.matching_type
        shape1 = layer_conf.input_dims[0]
        shape2 = layer_conf.input_dims[1]
        if self.matching_type == 'general':
            self.linear_in = nn.Linear(shape1[-1], shape2[-1], bias=False)

    def forward(self, string1, string1_len, string2, string2_len):
        """ process inputs

        Args:
            string1 (Tensor): [batch_size, seq_len1, dim]
            string1_len (Tensor): [batch_size]
            string2 (Tensor): [batch_size, seq_len2, dim]
            string2_len (Tensor): [batch_size]

        Returns:
            Tensor: [batch_size, seq_len1, seq_len2]

        """
        padded_seq_len1 = string1.shape[1]
        padded_seq_len2 = string2.shape[1]
        seq_dim1 = string1.shape[-1]
        seq_dim2 = string2.shape[-1]
        x1 = string1
        x2 = string2
        result = None
        if self.matching_type == 'dot' or self.matching_type == 'general':
            if self.matching_type == 'general':
                x1 = x1.view(-1, seq_dim1)
                x1 = self.linear_in(x1)
                x1 = x1.view(-1, padded_seq_len1, seq_dim2)
            result = torch.bmm(x1, x2.transpose(1, 2).contiguous())
            result = torch.unsqueeze(result, -1)
        else:
            if self.matching_type == 'mul':

                def func(x, y):
                    return x * y
            elif self.matching_type == 'plus':

                def func(x, y):
                    return x + y
            elif self.matching_type == 'minus':

                def func(x, y):
                    return x - y
            elif self.matching_type == 'concat':

                def func(x, y):
                    return torch.cat([x, y], dim=-1)
            else:
                raise ValueError(f'Invalid matching type.{self.matching_type} received.Mut be in `dot`, `general`, `mul`, `plus`, `minus` and `concat`.')
            x1_exp = torch.stack([x1] * padded_seq_len2, dim=2)
            x2_exp = torch.stack([x2] * padded_seq_len1, dim=1)
            result = func(x1_exp, x2_exp)
        return result, padded_seq_len1


class LinearAttention(BaseLayer):
    """  Linear attention.
    Combinate the original sequence along the sequence_length dimension.

    Args:
        layer_conf (LinearAttentionConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        """

        Args:
            layer_conf (LinearAttentionConf): configuration of a layer
        """
        super(LinearAttention, self).__init__(layer_conf)
        self.attention_weight = nn.Parameter(torch.FloatTensor(torch.randn(self.layer_conf.attention_weight_dim, 1)))

    def forward(self, string, string_len=None):
        """ process inputs

        Args:
            string (Variable): (batch_size, sequence_length, dim)
            string_len (ndarray or None): [batch_size]

        Returns:
            Variable:
                if keep_dim == False:
                    Output dimention: (batch_size, dim)
                else:
                    just reweight along the sequence_length dimension: (batch_size, sequence_length, dim)

        """
        attention_weight = torch.mm(string.contiguous().view(string.shape[0] * string.shape[1], string.shape[2]), self.attention_weight)
        attention_weight = nn.functional.softmax(attention_weight.view(string.shape[0], string.shape[1]), dim=1)
        attention_tiled = attention_weight.unsqueeze(2).expand_as(string)
        string_reweighted = torch.mul(string, attention_tiled)
        if self.layer_conf.keep_dim is False:
            string_reweighted = torch.sum(string_reweighted, 1)
        return string_reweighted, string_len


class MatchAttention(BaseLayer):
    """  MatchAttention layer for DrQA
    [paper]  https://arxiv.org/abs/1704.00051

    Given sequences X and Y, match sequence Y to each element in X.

    Args:
        layer_conf (MatchAttentionConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(MatchAttention, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]
        self.linear = nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1])
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x:      [batch_size, x_max_len, dim].
            x_len:  [batch_size], default is None.
            y:      [batch_size, y_max_len, dim].
            y_len:  [batch_size], default is None.

        Returns:
            output: has the same shape as x.

        """
        x_proj = self.linear(x)
        y_proj = self.linear(y)
        if self.activation:
            x_proj = self.activation(x_proj)
            y_proj = self.activation(y_proj)
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        alpha = self.softmax(scores)
        output = alpha.bmm(y)
        return output, x_len


class Seq2SeqAttention(BaseLayer):
    """ Linear layer

    Args:
        layer_conf (LinearConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Seq2SeqAttention, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.W = nn.Linear(layer_conf.input_dims[0][-1] * 3, 1)
        self.attention_dropout = nn.Dropout(layer_conf.attention_dropout)

    def forward(self, string, string_len, string2, string2_len=None):
        """ utilize both string2 and string itself to generate attention weights to represent string.
            There are two steps:
                1. get a string2 to string attention to represent string.
                2. get a string to string attention to represent string it self.
                3. merge the two representation above.

        Args:
            string (Variable): [batch_size, string_seq_len, dim].
            string_len (ndarray or None): [batch_size], default is None.
            string2 (Variable): [batch_size, string2_seq_len, dim].
            string2_len (ndarray or None): [batch_size], default is None.

        Returns:
            Variable: has the same shape as string.
        """
        assert string.size()[2] == string2.size()[2], 'The dimension of axis 2 of string and string2 must be consistent! But now, string.size() is %s and string2.size() is %s' % (string.size(), string2.size())
        batch_size = string.size()[0]
        string_seq_len = string.size()[1]
        string2_seq_len = string2.size()[1]
        feature_dim = string.size()[2]
        string2_aug = string2.unsqueeze(1).expand(batch_size, string_seq_len, string2_seq_len, feature_dim)
        string_aug = string.unsqueeze(1).expand(batch_size, string2_seq_len, string_seq_len, feature_dim)
        string2_aug = string2_aug.contiguous().view(batch_size * string_seq_len * string2.size()[1], feature_dim)
        string_aug = string_aug.contiguous().view(batch_size * string2_seq_len * string_seq_len, feature_dim)
        string2_string_comb = torch.cat((string2_aug, string_aug, string_aug * string2_aug), 1)
        attention = self.W(string2_string_comb)
        attention = self.attention_dropout(attention)
        attention = attention.view(batch_size, string_seq_len, string2_seq_len)
        string_to_string_att_weight = torch.unsqueeze(nn.Softmax(dim=1)(torch.max(attention, 2)[0]), 2)
        string_to_string_attention = string_to_string_att_weight * string
        string2_to_string_att_weight = nn.Softmax(dim=2)(attention)
        string2_to_string_attention = torch.sum(string2.unsqueeze(dim=1) * string2_to_string_att_weight.unsqueeze(dim=3), dim=2)
        string_out = torch.cat((string_to_string_attention, string2_to_string_attention), 2)
        return string_out, string_len


class LSTMCharEmbedding(BaseLayer):
    """
    This layer implements the character embedding use LSTM
    Args:
        layer_conf (LSTMCharEmbeddingConf): configuration of LSTMCharEmbedding
    """

    def __init__(self, layer_conf):
        super(LSTMCharEmbedding, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.char_embeddings = nn.Embedding(layer_conf.vocab_size, layer_conf.embedding_matrix_dim, padding_idx=self.layer_conf.padding)
        nn.init.uniform_(self.char_embeddings.weight, -0.001, 0.001)
        if layer_conf.bidirect_flag:
            self.dim = layer_conf.dim // 2
        self.dropout = nn.Dropout(layer_conf.dropout)
        self.char_lstm = nn.LSTM(layer_conf.embedding_matrix_dim, self.dim, num_layers=1, batch_first=True, bidirectional=layer_conf.bidirect_flag)
        if self.is_cuda():
            self.char_embeddings = self.char_embeddings
            self.dropout = self.dropout
            self.char_lstm = self.char_lstm

    def forward(self, string):
        """
        Step1: [batch_size, seq_len, char num in words] -> [batch_size*seq_len, char num in words]
        Step2: lookup embedding matrix -> [batch_size*seq_len, char num in words, embedding_dim]
        Step3: after lstm operation, got [num_layer* num_directions, batch_size * seq_len, dim]
        Step5: reshape -> [batch_size, seq_len, dim]

        Args:
            string (Variable): [[char ids of word1], [char ids of word2], [...], ...], shape: [batch_size, seq_len, char num in words]

        Returns:
            Variable: [batch_size, seq_len, output_dim]

        """
        string_reshaped = string.view(string.size()[0] * string.size()[1], -1)
        char_embs_lookup = self.char_embeddings(string_reshaped).float()
        char_embs_drop = self.dropout(char_embs_lookup)
        char_hidden = None
        char_rnn_out, char_hidden = self.char_lstm(char_embs_drop, char_hidden)
        string_out = char_hidden[0].transpose(1, 0).contiguous().view(string.size()[0], string.size()[1], -1)
        return string_out


def get_seq_mask(seq_len, max_seq_len=None):
    """

    Args:
        seq_len (ndarray): 1d numpy array/list

    Returns:
        ndarray : 2d array seq_len_mask. the mask symbol for a real token is 1 and for <pad> is 0.

    """
    if torch.is_tensor(seq_len):
        seq_len = seq_len.cpu().numpy()
    if max_seq_len is None:
        max_seq_len = seq_len.max()
    masks = np.array([([1] * seq_len[i] + [0] * (max_seq_len - seq_len[i])) for i in range(len(seq_len))])
    return masks


class SLUDecoder(BaseLayer):
    """ Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        layer_conf (SLUEncoderConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(SLUDecoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.embedding = nn.Embedding(layer_conf.decoder_vocab_size, layer_conf.decoder_emb_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.lstm = nn.LSTM(layer_conf.decoder_emb_dim + layer_conf.input_dims[0][-1] + layer_conf.input_context_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, batch_first=True)
        self.attn = nn.Linear(layer_conf.input_context_dims[0][-1], layer_conf.hidden_dim * layer_conf.num_layers)
        self.slot_out = nn.Linear(layer_conf.input_context_dims[0][-1] + layer_conf.hidden_dim * 1 * layer_conf.num_layers, layer_conf.decoder_vocab_size)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """

        Args:
            hidden : 1,B,D
            encoder_outputs : B,T,D
            encoder_maskings : B,T # ByteTensor
        """
        hidden = hidden.view(hidden.size()[1], -1).unsqueeze(2)
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        energies = energies.view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1000000000000.0)
        alpha = F.softmax(attn_energies)
        alpha = alpha.unsqueeze(1)
        context = alpha.bmm(encoder_outputs)
        return context

    def forward(self, string, string_len, context, encoder_outputs):
        """ process inputs

        Args:
            string (Variable): word ids, [batch_size, seq_len]
            string_len (ndarray): [batch_size]
            context (Variable): [batch_size, 1, input_dim]
            encoder_outputs (Variable): [batch_size, max_seq_len, input_dim]

        Returns:
            Variable : decode scores with shape [batch_size, seq_len, decoder_vocab_size]

        """
        batch_size = string.size(0)
        if torch.cuda.device_count() > 1:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len, max_seq_len=string.shape[1]))
        else:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len))
        decoded = torch.LongTensor([[1] * batch_size])
        hidden_init = torch.zeros(self.layer_conf.num_layers * 1, batch_size, self.layer_conf.hidden_dim)
        context_init = torch.zeros(self.layer_conf.num_layers * 1, batch_size, self.layer_conf.hidden_dim)
        if self.is_cuda():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            string_mask = string_mask
            decoded = decoded
            hidden_init = hidden_init
            context_init = context_init
        decoded = decoded.transpose(1, 0)
        embedded = self.embedding(decoded)
        hidden = hidden_init, context_init
        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        for i in range(length):
            aligned = aligns[i].unsqueeze(1)
            self.lstm.flatten_parameters()
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2), hidden)
            concated = torch.cat((hidden[0].view(1, batch_size, -1), context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            _, decoded = torch.max(softmaxed, 1)
            embedded = self.embedding(decoded.unsqueeze(1))
            context = self.Attention(hidden[0], encoder_outputs, string_mask)
        slot_scores = torch.cat(decode, 1)
        return slot_scores.view(batch_size, length, -1)


class SLUEncoder(BaseLayer):
    """ Spoken Language Understanding Encoder

    References:
        Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent detection and slot filling. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, (1), 685–689. https://doi.org/10.21437/Interspeech.2016-1352

    Args:
        layer_conf (SLUEncoderConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(SLUEncoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf
        self.lstm = nn.LSTM(layer_conf.input_dims[0][-1], layer_conf.hidden_dim, layer_conf.num_layers, batch_first=True, bidirectional=True, dropout=layer_conf.dropout)

    def forward(self, string, string_len):
        """ process inputs

        Args:
            string (Variable): [batch_size, seq_len, dim]
            string_len (ndarray): [batch_size]

        Returns:
            Variable: output of bi-lstm with shape [batch_size, seq_len, 2 * hidden_dim]
            ndarray: string_len with shape [batch_size]
            Variable: context with shape [batch_size, 1, 2 * hidden_dim]

        """
        if torch.cuda.device_count() > 1:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len, max_seq_len=string.shape[1]))
        else:
            string_mask = torch.ByteTensor(1 - get_seq_mask(string_len))
        hidden_init = torch.zeros(self.layer_conf.num_layers * 2, string.size(0), self.layer_conf.hidden_dim)
        context_init = torch.zeros(self.layer_conf.num_layers * 2, string.size(0), self.layer_conf.hidden_dim)
        if self.is_cuda():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            string_mask = string_mask
            hidden_init = hidden_init
            context_init = context_init
        hidden = hidden_init, context_init
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(string, hidden)
        assert output.shape[1] == string_mask.shape[1]
        real_context = []
        for i, o in enumerate(output):
            real_length = string_mask[i].data.tolist().count(0)
            real_context.append(o[real_length - 1])
        return output, torch.cat(real_context).view(string.size(0), -1).unsqueeze(1)


class Add2D(nn.Module):
    """ Add2D layer to get sum of two sequences(2D representation)

    Args:
        layer_conf (Add2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Add2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length Add2D layer returns is the length of first input')

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
        logging.warning('The length Add3D layer returns is the length of first input')

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
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] == 1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError('For layer Add3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal')
        return torch.add(args[0], args[2]), args[1]


class ElementWisedMultiply2D(nn.Module):
    """ ElementWisedMultiply2D layer to do Element-Wised Multiply of two sequences(2D representation)

    Args:
        layer_conf (ElementWisedMultiply2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(ElementWisedMultiply2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length ElementWisedMultiply2D layer returns is the length of first input')

    def forward(self, *args):
        """ process input

        Args:
            *args: (Tensor): string, string_len, string2, string2_len
                e.g. string (Tensor): [batch_size, dim], string_len (Tensor): [batch_size]


        Returns:
            Tensor: [batch_size, output_dim], [batch_size]
        """
        return torch.addcmul(torch.zeros(args[0].size()), 1, args[0], args[2]), args[1]


class ElementWisedMultiply3D(nn.Module):
    """ ElementWisedMultiply3D layer to do Element-Wised Multiply of two sequences(3D representation)

    Args:
        layer_conf (ElementWisedMultiply3DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(ElementWisedMultiply3D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length ElementWisedMultiply3D layer returns is the length of first input')

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
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] == 1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError('For layer ElementWisedMultiply3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal')
        return torch.addcmul(torch.zeros(args[0].size()), 1, args[0], args[2]), args[1]


class MatrixMultiply(nn.Module):
    """ MatrixMultiply layer to multiply two matrix

    Args:
        layer_conf (MatrixMultiplyConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(MatrixMultiply, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length MatrixMultiply layer returns is the length of first input')

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
                raise Exception('the dimensions of the two matrix for multiply is illegal')
        if self.layer_conf.operation == 'seq_based':
            if args[0].shape[1] == args[2].shape[1]:
                string = args[0].permute(0, 2, 1)
                return torch.matmul(string, args[2]), args[1]
            else:
                raise Exception('the dimensions of the two matrix for multiply is illegal')
        if self.layer_conf.operation == 'dim_based':
            if args[0].shape[2] == args[2].shape[2]:
                string = args[2].permute(0, 2, 1)
                return torch.matmul(args[0], string), args[1]
            else:
                raise Exception('the dimensions of the two matrix for multiply is illegal')


class Minus2D(nn.Module):
    """Minus2D layer to get subtraction of two sequences(2D representation)

    Args:
        layer_conf (Minus2DConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Minus2D, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length Minus2D layer returns is the length of first input')

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
        logging.warning('The length Minus3D layer returns is the length of first input')

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
        if args[0].shape[1] * args[0].shape[2] != args[2].shape[1] * args[2].shape[2]:
            if args[0].shape[1] == args[2].shape[1] and (input_dims[1][-1] == 1 or input_dims[0][-1] == 1):
                dim_flag = True
            else:
                dim_flag = False
        if dim_flag == False:
            raise ConfigurationError('For layer Minus3D, the dimensions of each inputs should be equal or 1 ,or the elements number of two inputs (expect for the first dimension) should be equal')
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


class CalculateDistance(BaseLayer):
    """ CalculateDistance layer to calculate the distance of sequences(2D representation)

    Args:
        layer_conf (CalculateDistanceConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(CalculateDistance, self).__init__(layer_conf)
        self.layer_conf = layer_conf

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x: [batch_size, dim]
            x_len: [batch_size]
            y: [batch_size, dim]
            y_len: [batch_size]
        Returns:
            Tensor: [batch_size, 1], None

        """
        batch_size = x.size()[0]
        if 'cos' in self.layer_conf.operations:
            result = F.cosine_similarity(x, y)
        elif 'euclidean' in self.layer_conf.operations:
            result = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        elif 'manhattan' in self.layer_conf.operations:
            result = torch.sum(torch.abs(x - y), dim=1)
        elif 'chebyshev' in self.layer_conf.operations:
            result = torch.abs(x - y).max(dim=1)
        else:
            raise ConfigurationError('This operation is not supported!')
        result = result.view(batch_size, 1)
        return result, None


class Combination(nn.Module):
    """ Combination layer to merge the representation of two sequence

    Args:
        layer_conf (CombinationConf): configuration of a layer
    """

    def __init__(self, layer_conf):
        super(Combination, self).__init__()
        self.layer_conf = layer_conf
        logging.warning('The length Combination layer returns is the length of first input')

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
        logging.warning('The length Concat2D layer returns is the length of first input')

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
        logging.warning('The length Concat3D layer returns is the length of first input')

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
                        raise Exception('Concat3D with axis = 2 require that the input sequences length should be the same!')
        return torch.cat(result, self.layer_conf.concat3D_axis), args[1]


class Expand_plus(BaseLayer):
    """  Expand_plus layer
    Given sequences X and Y, put X and Y expand_dim, and then add.

    Args:
        layer_conf (Expand_plusConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Expand_plus, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x:      [batch_size, x_max_len, dim].
            x_len:  [batch_size], default is None.
            y:      [batch_size, y_max_len, dim].
            y_len:  [batch_size], default is None.

        Returns:
            output: batch_size, x_max_len, y_max_len, dim].

        """
        x_new = torch.stack([x] * y.size()[1], 2)
        y_new = torch.stack([y] * x.size()[1], 1)
        return x_new + y_new, None


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


class Match(BaseLayer):
    """  Match layer
    Given sequences X and Y, match sequence Y to each element in X.

    Args:
        layer_conf (MatchConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(Match, self).__init__(layer_conf)
        assert layer_conf.input_dims[0][-1] == layer_conf.input_dims[1][-1]
        self.linear = nn.Linear(layer_conf.input_dims[0][-1], layer_conf.input_dims[0][-1])
        if layer_conf.activation:
            self.activation = eval('nn.' + self.layer_conf.activation)()
        else:
            self.activation = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len, y, y_len):
        """

        Args:
            x:      [batch_size, x_max_len, dim].
            x_len:  [batch_size], default is None.
            y:      [batch_size, y_max_len, dim].
            y_len:  [batch_size], default is None.

        Returns:
            output: has the same shape as x.

        """
        x_proj = self.linear(x)
        y_proj = self.linear(y)
        if self.activation:
            x_proj = self.activation(x_proj)
            y_proj = self.activation(y_proj)
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        return scores, x_len


class MLP(nn.Module):
    """ MLP layer

    Args:
        layer_conf (MLPConf): configuration of a layer

    """

    def __init__(self, layer_conf):
        super(MLP, self).__init__()
        self.layer_conf = layer_conf
        self.n_state = self.layer_conf.input_dims[0][-1]
        self.c_fc = nn.Linear(self.layer_conf.input_dims[0][-1], 4 * self.n_state)
        self.c_proj = nn.Linear(4 * self.n_state, self.layer_conf.input_dims[0][-1])

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.n_state % self.layer_conf.n_head == 0
        self.c_attn = nn.Linear(self.layer_conf.input_dims[0][-1], self.n_state * 3)
        self.c_proj = nn.Linear(self.layer_conf.input_dims[0][-1], self.n_state)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
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
        new_x_shape = x.size()[:-1] + (self.layer_conf.n_head, x.size(-1) // self.layer_conf.n_head)
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
        self.register_buffer('b', torch.tril(torch.ones(string.shape[1], string.shape[1])).view(1, 1, string.shape[1], string.shape[1]))
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
                new_tags[:, (idx)] = tags[:, (idx - 1)] * tag_size + tags[:, (idx)]
        end_transition = transitions[:, (crf_layer_conf.target_dict[crf_layer_conf.STOP_TAG])].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def forward(self, forward_score, scores, masks, tags, transitions, crf_layer_conf):
        """
        
        :param forward_score: Tensor scale
        :param scores: Tensor [seq_len, batch_size, target_size, target_size]
        :param masks:  Tensor [batch_size, seq_len]
        :param tags:   Tensor [batch_size, seq_len]
        :return: goal_score - forward_score
        """
        gold_score = self._score_sentence(scores, masks, tags, transitions, crf_layer_conf)
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


class Loss(nn.Module):
    """
    For support multi_task or multi_output, the loss type changes to list.
    Using class Loss for parsing and constructing the loss list.
    Args:
        loss_conf: the loss for multi_task or multi_output.
                   multi_loss_op: the operation for multi_loss
                   losses: list type. Each element is single loss.
                   eg: "loss": {
                            "multi_loss_op": "weighted_sum",
                            "losses": [
                              {
                                "type": "CrossEntropyLoss",
                                "conf": {
                                    "gamma": 0,
                                    "alpha": 0.5,
                                    "size_average": true
                                },
                                "inputs": ["start_output", "start_label"]
                              },
                              {
                                "type": "CrossEntropyLoss",
                                "conf": {
                                    "gamma": 0,
                                    "alpha": 0.5,
                                    "size_average": true
                                },
                                "inputs": ["end_output", "end_label"]
                              }
                            ],
                            "weights": [0.5, 0.5]
                        }
    """

    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.loss_fn = nn.ModuleList()
        self.loss_input = []
        self.weights = kwargs['weights'] if 'weights' in kwargs else None
        support_loss_op = set(LossOperationType.__members__.keys())
        if kwargs['multiLoss']:
            if not kwargs['multi_loss_op'].lower() in support_loss_op:
                raise Exception('The multi_loss_op %s is not supported. Supported multi_loss_op are: %s' % (kwargs['multi_loss_op'], support_loss_op))
            self.multi_loss_op = kwargs['multi_loss_op']
        for single_loss in kwargs['losses']:
            if not single_loss['inputs'][0] in kwargs['output_layer_id'] or not single_loss['inputs'][1] in kwargs['answer_column_name']:
                raise Exception('The loss inputs are excepted to be part of output_layer_id and targets!')
            self.loss_fn.append(eval(single_loss['type'])(**single_loss['conf']))
            self.loss_input.append(single_loss['inputs'])

    def forward(self, model_outputs, targets):
        """
        compute multi_loss according to multi_loss_op
        :param model_outputs: the representation of model output layer
                              :type: dict {output_layer_id: output layer data}
        :param targets: the label of raw data
                        :type: dict {target: data}
        :return:
        """
        all_losses = []
        result_loss = 0.0
        for index, single_loss_fn in enumerate(self.loss_fn):
            all_losses.append(single_loss_fn(model_outputs[self.loss_input[index][0]], targets[self.loss_input[index][1]]))
        if hasattr(self, 'multi_loss_op'):
            if LossOperationType[self.multi_loss_op.lower()] == LossOperationType.weighted_sum:
                for index, single_loss in enumerate(all_losses):
                    result_loss += self.weights[index] * single_loss
        else:
            result_loss = all_losses[0]
        return result_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseLayer,
     lambda: ([], {'layer_conf': 1}),
     lambda: ([], {}),
     False),
    (Flatten,
     lambda: ([], {'layer_conf': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ForgetMult,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (QRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (QRNNLayer,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_microsoft_NeuronBlocks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

