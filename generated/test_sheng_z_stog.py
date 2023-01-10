import sys
_module = sys.modules[__name__]
del sys
stog = _module
algorithms = _module
dict_merge = _module
maximum_spanning_tree = _module
commands = _module
evaluate = _module
predict = _module
subcommand = _module
train = _module
data = _module
dataset = _module
dataset_builder = _module
dataset_readers = _module
abstract_meaning_representation = _module
amr_parsing = _module
amr = _module
amr_concepts = _module
date = _module
entity = _module
ordinal = _module
polarity = _module
polite = _module
quantity = _module
score = _module
url = _module
graph_repair = _module
io = _module
node_utils = _module
postprocess = _module
expander = _module
node_restore = _module
wikification = _module
preprocess = _module
feature_annotator = _module
input_cleaner = _module
morph = _module
recategorizer = _module
sense_remover = _module
text_anonymizor = _module
propbank_reader = _module
dataset_reader = _module
fields = _module
adjacency_field = _module
array_field = _module
field = _module
index_field = _module
knowledge_graph_field = _module
label_field = _module
list_field = _module
metadata_field = _module
multilabel_field = _module
production_rule_field = _module
sequence_field = _module
sequence_label_field = _module
span_field = _module
text_field = _module
instance = _module
iterators = _module
basic_iterator = _module
bucket_iterator = _module
data_iterator = _module
epoch_tracking_bucket_iterator = _module
multiprocess_iterator = _module
token_indexers = _module
dep_label_indexer = _module
elmo_indexer = _module
ner_tag_indexer = _module
openai_transformer_byte_pair_indexer = _module
pos_tag_indexer = _module
single_id_token_indexer = _module
token_characters_indexer = _module
token_indexer = _module
tokenizers = _module
bert_tokenizer = _module
character_tokenizer = _module
token = _module
tokenizer = _module
word_filter = _module
word_splitter = _module
word_stemmer = _module
word_tokenizer = _module
vocabulary = _module
metrics = _module
attachment_score = _module
metric = _module
seq2seq_metrics = _module
models = _module
model = _module
stog = _module
modules = _module
attention = _module
biaffine_attention = _module
dot_production_attention = _module
mlp_attention = _module
attention_layers = _module
global_attention = _module
augmented_lstm = _module
decoders = _module
deep_biaffine_graph_decoder = _module
generator = _module
pointer_generator = _module
rnn_decoder = _module
encoder_base = _module
initializers = _module
input_variational_dropout = _module
linear = _module
bilinear = _module
optimizer = _module
seq2seq_encoders = _module
pytorch_seq2seq_wrapper = _module
seq2seq_bert_encoder = _module
seq2seq_encoder = _module
seq2vec_encoders = _module
boe_encoder = _module
cnn_encoder = _module
pytorch_seq2vec_wrapper = _module
seq2vec_encoder = _module
stacked_bilstm = _module
stacked_lstm = _module
text_field_embedders = _module
basic_text_field_embedder = _module
text_field_embedder = _module
time_distributed = _module
token_embedders = _module
embedding = _module
openai_transformer_embedder = _module
token_characters_encoder = _module
token_embedder = _module
predictors = _module
predictor = _module
training = _module
tensorboard = _module
trainer = _module
utils = _module
archival = _module
checks = _module
environment = _module
exception_hook = _module
extract_tokens_from_amr = _module
file = _module
from_params = _module
logging = _module
nn = _module
params = _module
registrable = _module
string = _module
time = _module
tqdm = _module

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


from typing import List


from typing import Iterator


from typing import Optional


import re


import logging


from collections import defaultdict


from typing import Dict


from typing import Union


from typing import Iterable


import numpy


from typing import Set


from typing import Tuple


from typing import Generic


from typing import TypeVar


from typing import Callable


from typing import Sequence


from typing import cast


from typing import MutableMapping


import itertools


import math


import random


from torch.multiprocessing import Manager


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


from torch.multiprocessing import get_logger


import torch.nn as nn


from torch.nn.parameter import Parameter


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import PackedSequence


import numpy as np


import copy


from typing import Type


import torch.nn.init


import torch.optim as optim


from torch.nn.utils import clip_grad_norm_


from torch.nn import Conv1d


from torch.nn import Linear


import warnings


from typing import IO


from typing import Any


from typing import NamedTuple


from torch.nn.functional import embedding


import time


from torch import cuda


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


DataArray = TypeVar('DataArray', torch.Tensor, Dict[str, torch.Tensor])


DEFAULT_NON_PADDED_NAMESPACES = '*tags', '*labels'


DEFAULT_OOV_TOKEN = '@@UNKNOWN@@'


DEFAULT_PADDING_TOKEN = '@@PADDING@@'


NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if isinstance(dct.get(k), dict) and isinstance(v, collections.Mapping):
            dict_merge(dct[k], v)
        else:
            dct[k] = v


class Params(object):
    """
    Parameters
    """

    def __init__(self, params):
        self.params = params

    def __eq__(self, other):
        if not isinstance(other, Params):
            logger.info('The params you compare is not an instance of Params. ({} != {})'.format(type(self), type(other)))
            return False
        this_flat_params = self.as_flat_dict()
        other_flat_params = other.as_flat_dict()
        if len(this_flat_params) != len(other_flat_params):
            logger.info('The numbers of parameters are different: {} != {}'.format(len(this_flat_params), len(other_flat_params)))
            return False
        same = True
        for k, v in this_flat_params.items():
            if k == 'environment.recover':
                continue
            if k not in other_flat_params:
                logger.info('The parameter "{}" is not specified.'.format(k))
                same = False
            elif other_flat_params[k] != v:
                logger.info('The values of "{}" not not the same: {} != {}'.format(k, v, other_flat_params[k]))
                same = False
        return same

    def __getitem__(self, item):
        if item in self.params:
            return self.params[item]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def items(self):
        return self.params.items()

    def get(self, key, default=None):
        return self.params.get(key, default)

    def as_flat_dict(self):
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value
        recurse(self.params, [])
        return flat_params

    def to_file(self, output_json_file):
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent='\t')

    @classmethod
    def from_file(cls, params_file_list):
        params_file_list = params_file_list.split(',')
        params_dict = {}
        for params_file in params_file_list:
            with open(params_file, encoding='utf-8') as f:
                if params_file.endswith('.yaml'):
                    dict_merge.dict_merge(params_dict, yaml.load(f))
                elif params_file.endswith('.json'):
                    params_dict = json.load(f)
                else:
                    raise NotImplementedError
        return cls(params_dict)

    def __repr__(self):
        return json.dumps(self.params, indent=2)

    def duplicate(self) ->'Params':
        """
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return Params(copy.deepcopy(self.params))


class Tqdm:
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) ->None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) ->None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {'mininterval': Tqdm.default_mininterval, **kwargs}
        return _tqdm(*args, **new_kwargs)


def namespace_match(pattern: str, namespace: str):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/common/util.py#L164

    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <https://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the ``defaultdict``), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of ``non_padded_namespaces``.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_namespaces`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_namespaces : ``Iterable[str]``
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that namespace, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(self, non_padded_namespaces: Iterable[str], padded_function: Callable[[], Any], non_padded_function: Callable[[], Any]) ->None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        self._non_padded_namespaces.update(non_padded_namespaces)


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):

    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) ->None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_namespaces, lambda : {(0): padding_token, (1): oov_token}, lambda : {})


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):

    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) ->None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces, lambda : {padding_token: 0, oov_token: 1}, lambda : {})


def _read_pretrained_tokens(embeddings_file_uri: str) ->List[str]:
    logger.info('Reading pretrained tokens from: %s', embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                logger.warning(f'Skipping line number %d: %s', line_number, line_begin)
    return tokens


def pop_max_vocab_size(params: Params) ->Union[int, Dict[str, int]]:
    """
    max_vocab_size is allowed to be either an int or a Dict[str, int] (or nothing).
    But it could also be a string representing an int (in the case of environment variable
    substitution). So we need some complex logic to handle it.
    """
    size = params.pop('max_vocab_size', None)
    if isinstance(size, Params):
        return size.as_dict()
    elif size is not None:
        try:
            return int(size)
        except:
            return size
    else:
        return None


class Vocabulary:
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    Parameters
    ----------
    counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
        anything.
    min_count : ``Dict[str, int]``, optional (default=None)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.
    max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_namespaces : ``Iterable[str]``, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is ``("*tags", "*labels")``, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.
    pretrained_files : ``Dict[str, str]``, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of ``only_include_pretrained_words``.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.
    min_pretrained_embeddings : ``Dict[str, int]``, optional
        If provided, specifies for each namespace a mininum number of lines (typically the
        most common words) to keep from pretrained embedding files, even for words not
        appearing in the data.
    only_include_pretrained_words : ``bool``, optional (default=False)
        This defines the stategy for using any pretrained embedding files which may have been
        specified in ``pretrained_files``. If False, an inclusive stategy is used: and words
        which are in the ``counter`` and in the pretrained file are added to the ``Vocabulary``,
        regardless of whether their count exceeds ``min_count`` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least ``min_count``).
    tokens_to_add : ``Dict[str, List[str]]``, optional (default=None)
        If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.
    """
    default_implementation = 'default'

    def __init__(self, counter: Dict[str, Dict[str, int]]=None, min_count: Dict[str, int]=None, max_vocab_size: Union[int, Dict[str, int]]=None, non_padded_namespaces: Iterable[str]=DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]]=None, only_include_pretrained_words: bool=False, tokens_to_add: Dict[str, List[str]]=None, min_pretrained_embeddings: Dict[str, int]=None) ->None:
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        self._extend(counter, min_count, max_vocab_size, non_padded_namespaces, pretrained_files, only_include_pretrained_words, tokens_to_add, min_pretrained_embeddings)

    def save_to_files(self, directory: str) ->None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.

        Parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning('vocabulary serialization directory %s is not empty', directory)
        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                None
        for namespace, mapping in self._index_to_token.items():
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    None

    @classmethod
    def from_files(cls, directory: str) ->'Vocabulary':
        """
        Loads a ``Vocabulary`` that was serialized using ``save_to_files``.

        Parameters
        ----------
        directory : ``str``
            The directory containing the serialized vocabulary.
        """
        logger.info('Loading token dictionary from %s.', directory)
        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
            non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]
        vocab = cls(non_padded_namespaces=non_padded_namespaces)
        for namespace_filename in os.listdir(directory):
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            namespace = namespace_filename.replace('.txt', '')
            if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                is_padded = False
            else:
                is_padded = True
            filename = os.path.join(directory, namespace_filename)
            vocab.set_from_file(filename, is_padded, namespace=namespace)
        return vocab

    def set_from_file(self, filename: str, is_padded: bool=True, oov_token: str=DEFAULT_OOV_TOKEN, namespace: str='tokens'):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        Parameters
        ----------
        filename : ``str``
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if ``is_padded``, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : ``bool``, optional (default=True)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            ``True``; while for tag or label vocabularies, this should typically be ``False``.  If
            ``True``, we add a padding token with index 0, and we enforce that the ``oov_token`` is
            present in the file.
        oov_token : ``str``, optional (default=DEFAULT_OOV_TOKEN)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            ``oov_token`` with ``self._oov_token``, because we only use one OOV token across
            namespaces.
        namespace : ``str``, optional (default="tokens")
            What namespace should we overwrite with this vocab file?
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {(0): self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            lines = input_file.read().split('\n')
            if lines and lines[-1] == '':
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index = i + 1 if is_padded else i
                token = line.replace('@@NEWLINE@@', '\n')
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], 'OOV token not found!'

    @classmethod
    def from_instances(cls, instances: Iterable['adi.Instance'], min_count: Dict[str, int]=None, max_vocab_size: Union[int, Dict[str, int]]=None, non_padded_namespaces: Iterable[str]=DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]]=None, only_include_pretrained_words: bool=False, tokens_to_add: Dict[str, List[str]]=None, min_pretrained_embeddings: Dict[str, int]=None) ->'Vocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.
        """
        logger.info('Fitting token dictionary from dataset.')
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda : defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        return cls(counter=namespace_token_counts, min_count=min_count, max_vocab_size=max_vocab_size, non_padded_namespaces=non_padded_namespaces, pretrained_files=pretrained_files, only_include_pretrained_words=only_include_pretrained_words, tokens_to_add=tokens_to_add, min_pretrained_embeddings=min_pretrained_embeddings)

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance']=None):
        """
        There are two possible ways to build a vocabulary; from a
        collection of instances, using :func:`Vocabulary.from_instances`, or
        from a pre-saved vocabulary, using :func:`Vocabulary.from_files`.
        You can also extend pre-saved vocabulary with collection of instances
        using this method. This method wraps these options, allowing their
        specification from a ``Params`` object, generated from a JSON
        configuration file.

        Parameters
        ----------
        params: Params, required.
        instances: Iterable['adi.Instance'], optional
            If ``params`` doesn't contain a ``directory_path`` key,
            the ``Vocabulary`` can be built directly from a collection of
            instances (i.e. a dataset). If ``extend`` key is set False,
            dataset instances will be ignored and final vocabulary will be
            one loaded from ``directory_path``. If ``extend`` key is set True,
            dataset instances will be used to extend the vocabulary loaded
            from ``directory_path`` and that will be final vocabulary used.

        Returns
        -------
        A ``Vocabulary``.
        """
        vocab_type = params.pop('type', None)
        if vocab_type is not None:
            return cls.by_name(vocab_type).from_params(params=params, instances=instances)
        extend = params.pop('extend', False)
        vocabulary_directory = params.pop('directory_path', None)
        if not vocabulary_directory and not instances:
            raise ConfigurationError('You must provide either a Params object containing a vocab_directory key or a Dataset to build a vocabulary from.')
        if extend and not instances:
            raise ConfigurationError("'extend' is true but there are not instances passed to extend.")
        if extend and not vocabulary_directory:
            raise ConfigurationError("'extend' is true but there is not 'directory_path' to extend from.")
        if vocabulary_directory and instances:
            if extend:
                logger.info('Loading Vocab from files and extending it with dataset.')
            else:
                logger.info('Loading Vocab from files instead of dataset.')
        if vocabulary_directory:
            vocab = Vocabulary.from_files(vocabulary_directory)
            if not extend:
                params.assert_empty('Vocabulary - from files')
                return vocab
        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab
        min_count = params.pop('min_count', None)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop('non_padded_namespaces', DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop('pretrained_files', {})
        min_pretrained_embeddings = params.pop('min_pretrained_embeddings', None)
        only_include_pretrained_words = params.pop_bool('only_include_pretrained_words', False)
        tokens_to_add = params.pop('tokens_to_add', None)
        params.assert_empty('Vocabulary - from dataset')
        return Vocabulary.from_instances(instances=instances, min_count=min_count, max_vocab_size=max_vocab_size, non_padded_namespaces=non_padded_namespaces, pretrained_files=pretrained_files, only_include_pretrained_words=only_include_pretrained_words, tokens_to_add=tokens_to_add, min_pretrained_embeddings=min_pretrained_embeddings)

    def _extend(self, counter: Dict[str, Dict[str, int]]=None, min_count: Dict[str, int]=None, max_vocab_size: Union[int, Dict[str, int]]=None, non_padded_namespaces: Iterable[str]=DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]]=None, only_include_pretrained_words: bool=False, tokens_to_add: Dict[str, List[str]]=None, min_pretrained_embeddings: Dict[str, int]=None) ->None:
        """
        This method can be used for extending already generated vocabulary.
        It takes same parameters as Vocabulary initializer. The token2index
        and indextotoken mappings of calling vocabulary will be retained.
        It is an inplace operation so None will be returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda : int_max_vocab_size)
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}
        self._retained_counter = counter
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}
        for namespace in (current_namespaces & extension_namespaces):
            original_padded = not any(namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces)
            extension_padded = not any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces)
            if original_padded != extension_padded:
                raise ConfigurationError('Common namespace {} has conflicting '.format(namespace) + 'setting of padded = True/False. ' + 'Hence extension cannot be done.')
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)
        for namespace in counter:
            if namespace in pretrained_files:
                pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0:
                    tokens_old = tokens_to_add.get(namespace, [])
                    tokens_new = pretrained_list[:min_embeddings]
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            else:
                pretrained_set = None
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)
        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def extend_from_instances(self, params: Params, instances: Iterable['adi.Instance']=()) ->None:
        """
        Extends an already generated vocabulary using a collection of instances.
        """
        min_count = params.pop('min_count', None)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop('non_padded_namespaces', DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop('pretrained_files', {})
        min_pretrained_embeddings = params.pop('min_pretrained_embeddings', None)
        only_include_pretrained_words = params.pop_bool('only_include_pretrained_words', False)
        tokens_to_add = params.pop('tokens_to_add', None)
        params.assert_empty('Vocabulary - from dataset')
        logger.info('Fitting token dictionary from dataset.')
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda : defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        self._extend(counter=namespace_token_counts, min_count=min_count, max_vocab_size=max_vocab_size, non_padded_namespaces=non_padded_namespaces, pretrained_files=pretrained_files, only_include_pretrained_words=only_include_pretrained_words, tokens_to_add=tokens_to_add, min_pretrained_embeddings=min_pretrained_embeddings)

    def is_padded(self, namespace: str) ->bool:
        """
        Returns whether or not there are padding and OOV tokens added to the given namepsace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str='tokens') ->int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError('Vocabulary tokens must be strings, or saving and loading will break.  Got %s (with type %s)' % (repr(token), type(token)))
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str='tokens') ->Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str='tokens') ->Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str='tokens') ->int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error('Namespace: %s', namespace)
                logger.error('Token: %s', token)
                raise

    def get_token_from_index(self, index: int, namespace: str='tokens') ->str:
        return self._index_to_token[namespace][index]

    def get_tokens_from_list(self, t, namespace):
        return [self.get_token_from_index(i, namespace) for i in t]

    def get_vocab_size(self, namespace: str='tokens') ->int:
        return len(self._token_to_index[namespace])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) ->str:
        base_string = f'Vocabulary with namespaces:\n'
        non_padded_namespaces = f'\tNon Padded Namespaces: {self._non_padded_namespaces}\n'
        namespaces = [f'\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n' for name in self._index_to_token]
        return ' '.join([base_string, non_padded_namespaces] + namespaces)

    def print_statistics(self) ->None:
        if self._retained_counter:
            logger.info("Printed vocabulary statistics are only for the part of the vocabulary generated from instances. If vocabulary is constructed by extending saved vocabulary with dataset instances, the directly loaded portion won't be considered here.")
            None
            for namespace in self._retained_counter:
                tokens_with_counts = list(self._retained_counter[namespace].items())
                tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
                None
                for token, freq in tokens_with_counts[:10]:
                    None
                tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)
                None
                for token, freq in tokens_with_counts[:10]:
                    None
                None
                for token, freq in reversed(tokens_with_counts[-10:]):
                    None
        else:
            logger.info('Vocabulary statistics cannot be printed since dataset instances were not used for its construction.')


class Field(Generic[DataArray]):
    """
    A ``Field`` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.

    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into tensors.  The ``Field`` API has methods around both of these steps, though they
    may not be needed for some concrete ``Field`` classes - if your field doesn't have any strings
    that need indexing, you don't need to implement ``count_vocab_items`` or ``index``.  These
    methods ``pass`` by default.

    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual tensors.
    """

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.

        A note on this ``counter``: because ``Fields`` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a ``TextField`` from sharing the same ids with labels in a ``LabelField`` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)

        Additionally, a single ``Field`` might want to use multiple namespaces - ``TextFields`` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.

        Because of this, the first key in the ``counter`` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the ``Field`` object, it does not return anything.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) ->Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like {'num_tokens': 13}.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def as_tensor(self, padding_lengths: Dict[str, int]) ->DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        torch Tensor (or a more complex data structure) of the correct shape.  We also take a
        couple of parameters that are important when constructing torch Tensors.

        Parameters
        ----------
        padding_lengths : ``Dict[str, int]``
            This dictionary will have the same keys that were produced in
            :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
            relevant dimension, aggregated across all instances in a batch.
        """
        raise NotImplementedError

    def empty_field(self) ->'Field':
        """
        So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
        option ``TextFields``), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
        ``count_vocab_items``, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in ``TextField``).
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: List[DataArray]) ->DataArray:
        """
        Takes the output of ``Field.as_tensor()`` from a list of ``Instances`` and merges it into
        one batched tensor for this ``Field``.  The default implementation here in the base class
        handles cases where ``as_tensor`` returns a single torch tensor per instance.  If your
        subclass returns something other than this, you need to override this method.

        This operation does not modify ``self``, but in some cases we need the information
        contained in ``self`` in order to perform the batching, so this is an instance method, not
        a class method.
        """
        return torch.stack(tensor_list)


class Instance:
    """
    An ``Instance`` is a collection of :class:`~stog.data.fields.field.Field` objects,
    specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will be indexed, after which multiple instances can be combined
    into a ``Batch`` and then converted into padded arrays.

    Parameters
    ----------
    fields : ``Dict[str, Field]``
        The ``Field`` objects that will be used to produce data arrays for this instance.
    """

    def __init__(self, fields: MutableMapping[str, Field]) ->None:
        self.fields = fields
        self.indexed = False

    def add_field(self, field_name: str, field: Field, vocab: Vocabulary=None) ->None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[field_name] = field
        if self.indexed:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary items in all of the
        ``Fields`` in this ``Instance``.
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) ->None:
        """
        Indexes all fields in this ``Instance`` using the provided ``Vocabulary``.
        This `mutates` the current object, it does not return a new ``Instance``.
        A ``DataIterator`` will call this on each pass through a dataset; we use the ``indexed``
        flag to make sure that indexing only happens once.

        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def get_padding_lengths(self) ->Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name.  Each ``Field`` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(self, padding_lengths: Dict[str, Dict[str, int]]=None) ->Dict[str, DataArray]:
        """
        Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of torch tensors for each field.

        If ``padding_lengths`` is omitted, we will call ``self.get_padding_lengths()`` to get the
        sizes of the tensors to create.
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors

    def __str__(self) ->str:
        base_string = f'Instance with fields:\n'
        return ' '.join([base_string] + [f'\t {name}: {field} \n' for name, field in self.fields.items()])


A = TypeVar('A')


def ensure_list(iterable: Iterable[A]) ->List[A]:
    """
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)


class Batch(Iterable):
    """
    A batch of Instances. In addition to containing the instances themselves,
    it contains helper functions for converting the data into tensors.
    """

    def __init__(self, instances: Iterable[Instance]) ->None:
        """
        A Batch just takes an iterable of instances in its constructor and hangs onto them
        in a list.
        """
        super().__init__()
        self.instances: List[Instance] = ensure_list(instances)
        self._check_types()

    def _check_types(self) ->None:
        """
        Check that all the instances have the same types.
        """
        all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__ for k, v in x.fields.items()} for x in self.instances]
        if not all([(all_instance_fields_and_types[0] == x) for x in all_instance_fields_and_types]):
            raise ConfigurationError('You cannot construct a Batch with non-homogeneous Instances.')

    def get_padding_lengths(self) ->Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this batch.  Each ``Instance``
        has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this batch into arrays of consistent length, or to set
        model parameters, etc.
        """
        padding_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        all_instance_lengths: List[Dict[str, Dict[str, int]]] = [instance.get_padding_lengths() for instance in self.instances]
        if not all_instance_lengths:
            return {**padding_lengths}
        all_field_lengths: Dict[str, List[Dict[str, int]]] = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)
        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x[padding_key] if padding_key in x else 0 for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value
        return {**padding_lengths}

    def as_tensor_dict(self, padding_lengths: Dict[str, Dict[str, int]]=None, verbose: bool=False) ->Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        This method converts this ``Batch`` into a set of pytorch Tensors that can be passed
        through a model.  In order for the tensors to be valid tensors, all ``Instances`` in this
        batch need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the tensors for each field in each instance into a set of
        batched tensors for each field.

        Parameters
        ----------
        padding_lengths : ``Dict[str, Dict[str, int]]``
            If a key is present in this dictionary with a non-``None`` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").
        verbose : ``bool``, optional (default=``False``)
            Should we output logging information when we're doing this padding?  If the batch is
            large, this is nice to have, because padding a large batch could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious (and really slow).

        Returns
        -------
        tensors : ``Dict[str, DataArray]``
            A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
            This is a `batch` of instances, so, e.g., if the instances have a "question" field and
            an "answer" field, the "question" fields for all of the instances will be grouped
            together into a single tensor, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of tensors, for batched computation. Additionally,
            for complex ``Fields``, the value of the dictionary key is not necessarily a single
            tensor.  For example, with the ``TextField``, the output is a dictionary mapping
            ``TokenIndexer`` keys to tensors. The number of elements in this sub-dictionary
            therefore corresponds to the number of ``TokenIndexers`` used to index the
            ``TextField``.  Each ``Field`` class is responsible for batching its own output.
        """
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        if verbose:
            logger.info('Padding batch of size %d to lengths %s', len(self.instances), str(padding_lengths))
            logger.info('Getting max lengths from instances')
        instance_padding_lengths = self.get_padding_lengths()
        if verbose:
            logger.info('Instance max lengths: %s', str(instance_padding_lengths))
        lengths_to_use: Dict[str, Dict[str, int]] = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths.keys():
                if padding_lengths[field_name].get(padding_key) is not None:
                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]
        field_tensors: Dict[str, list] = defaultdict(list)
        if verbose:
            logger.info('Now actually padding instances to length: %s', str(lengths_to_use))
        for instance in self.instances:
            for field, tensors in instance.as_tensor_dict(lengths_to_use).items():
                field_tensors[field].append(tensors)
        field_classes = self.instances[0].fields
        final_fields = {}
        for field_name, field_tensor_list in field_tensors.items():
            final_fields[field_name] = field_classes[field_name].batch_tensors(field_tensor_list)
        return final_fields

    def __iter__(self) ->Iterator[Instance]:
        return iter(self.instances)

    def index_instances(self, vocab: Vocabulary) ->None:
        for instance in self.instances:
            instance.index_fields(vocab)

    def print_statistics(self) ->None:
        sequence_field_lengths: Dict[str, List] = defaultdict(list)
        for instance in self.instances:
            if not instance.indexed:
                raise ConfigurationError('Instances must be indexed with vocabulary before asking to print dataset statistics.')
            for field, field_padding_lengths in instance.get_padding_lengths().items():
                for key, value in field_padding_lengths.items():
                    sequence_field_lengths[f'{field}.{key}'].append(value)
        None
        for name, lengths in sequence_field_lengths.items():
            None
            None
        None
        for i in list(numpy.random.randint(len(self.instances), size=10)):
            None
            None


_DEFAULT_WEIGHTS = 'best.th'


def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) ->torch.Storage:
        if cuda_device >= 0:
            return storage
        else:
            return storage
    return inner_device_mapping


def get_device_of(tensor: torch.Tensor) ->int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def has_tensor(obj) ->bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """
    if not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def remove_pretrained_embedding_params(params):

    def recurse(parameters, key):
        for k, v in parameters.items():
            if key == k:
                parameters[key] = None
            elif isinstance(v, dict):
                recurse(v, key)
    recurse(params, 'pretrained_file')


class Model(torch.nn.Module):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.
    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.
    In order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.
    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    :class:`~allennlp.training.Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by :class:`~allennlp.training.Trainer`.
    """
    _warn_for_unseparable_batches: Set[str] = set()

    def __init__(self, regularizer=None) ->None:
        super().__init__()
        self._regularizer = regularizer

    def get_regularization_penalty(self) ->Union[float, torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            return 0.0
        else:
            return self._regularizer(self)

    def get_parameters_for_histogram_tensorboard_logging(self) ->List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        return [name for name, _ in self.named_parameters()]

    def forward(self, inputs) ->Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.
        The input is comprised of everything required to perform a
        training update, `including` labels - you define the signature here!
        It is down to the user to ensure that inference can be performed
        without the presence of these labels. Hence, any inputs not available at
        inference time should only be used inside a conditional block.
        The intended sketch of this method is as follows::
            def forward(self, input1, input2, targets=None):
                ....
                ....
                output1 = self.layer1(input1)
                output2 = self.layer2(input2)
                output_dict = {"output1": output1, "output2": output2}
                if targets is not None:
                    # Function returning a scalar torch.Tensor, defined by the user.
                    loss = self._compute_loss(output1, output2, targets)
                    output_dict["loss"] = loss
                return output_dict
        Parameters
        ----------
        inputs:
            Tensors comprising everything needed to perform a training update, `including` labels,
            which should be optional (i.e have a default value of ``None``).  At inference time,
            simply pass the relevant inputs, not including the labels.
        Returns
        -------
        output_dict: ``Dict[str, torch.Tensor]``
            The outputs from the model. In order to train a model using the
            :class:`~allennlp.training.Trainer` api, you must provide a "loss" key pointing to a
            scalar ``torch.Tensor`` representing the loss to be optimized.
        """
        raise NotImplementedError

    def forward_on_instance(self, instance) ->Dict[str, numpy.ndarray]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        raise NotImplementedError

    def forward_on_instances(self, instances) ->List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.
        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.
        cuda_device : int, required
            The GPU device to use.  -1 means use the CPU.
        Returns
        -------
        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = move_to_device(dataset.as_tensor_dict(), device)
            outputs = self.decode(self(model_input))
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                outputs[name] = output
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def decode(self, output_dict: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        """
        Takes the result of :func:`forward` and runs inference / decoding / whatever
        post-processing you need to do your model.  The intent is that ``model.forward()`` should
        produce potentials or probabilities, and then ``model.decode()`` can take those results and
        run some kind of beam search or constrained inference or whatever is necessary.  This does
        not handle all possible decoding use cases, but it at least handles simple kinds of
        decoding.
        This method `modifies` the input dictionary, and also `returns` the same dictionary.
        By default in the base class we do nothing.  If your model has some special decoding step,
        override this method.
        """
        return output_dict

    def get_metrics(self, reset: bool=False) ->Dict[str, float]:
        """
        Returns a dictionary of metrics. This method will be called by
        :class:`allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialization.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
        should be populated during the call to ``forward``, with the
        :class:`~allennlp.training.Metric` handling the accumulation of the metric until this
        method is called.
        """
        return {}

    def _get_prediction_device(self):
        """
        This method checks the device of the model parameters to determine the cuda_device
        this model should be run on for predictions.  If there are no parameters, it returns -1.
        Returns
        -------
        The cuda device this model should run on for predictions.
        """
        devices = {get_device_of(param) for param in self.parameters()}
        if len(devices) > 1:
            devices_string = ', '.join(str(x) for x in devices)
            raise ConfigurationError(f'Parameters have mismatching cuda_devices: {devices_string}')
        elif len(devices) == 1 and all(i >= 0 for i in devices):
            device = torch.device('cuda:{}'.format(devices.pop()))
        else:
            device = torch.device('cpu')
        return device

    def _maybe_warn_for_unseparable_batches(self, output_key: str):
        """
        This method warns once if a user implements a model which returns a dictionary with
        values which we are unable to split back up into elements of the batch. This is controlled
        by a class attribute ``_warn_for_unseperable_batches`` because it would be extremely verbose
        otherwise.
        """
        if output_key not in self._warn_for_unseparable_batches:
            logger.warning(f"Encountered the {output_key} key in the model's return dictionary which couldn't be split by the batch size. Key will be ignored.")
            self._warn_for_unseparable_batches.add(output_key)

    def set_vocab(self, vocab):
        self.vocab = vocab

    @classmethod
    def _load(cls, config: Params, serialization_dir: str, weights_file: str=None, device=None) ->'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)
        vocab_dir = os.path.join(serialization_dir, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)
        model_params = config['model']
        remove_pretrained_embedding_params(model_params)
        model = cls.from_params(vocab=vocab, params=model_params)
        model_state = torch.load(weights_file, map_location=device_mapping(-1))
        if not isinstance(model, torch.nn.DataParallel):
            model_state = {re.sub('^module\\.', '', k): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.set_vocab(vocab)
        model
        return model

    @classmethod
    def load(cls, config: Params, serialization_dir: str, weights_file: str=None, device=None) ->'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        model_type = config['model']['model_type']
        return getattr(Models, model_type)._load(config, serialization_dir, weights_file, device)


class BiaffineAttention(nn.Module):
    """
    Adopted from NeuroNLP2:
        https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/nn/modules/attention.py

    Bi-Affine attention layer.
    """

    def __init__(self, input_size_encoder, input_size_decoder, num_labels=1, biaffine=True, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(BiaffineAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.W_e)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        """
        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]
        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)
        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b
        if mask_d is not None and mask_e is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)
        return output


class Token:
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, dependency relation, and similar information.  These fields match spacy's
    exactly, so we can just use a spacy token for this.

    Parameters
    ----------
    text : ``str``, optional
        The original text represented by this token.
    idx : ``int``, optional
        The character offset of this token into the tokenized passage.
    lemma : ``str``, optional
        The lemma of this token.
    pos : ``str``, optional
        The coarse-grained part of speech of this token.
    tag : ``str``, optional
        The fine-grained part of speech of this token.
    dep : ``str``, optional
        The dependency relation for this token.
    ent_type : ``str``, optional
        The entity type (i.e., the NER tag) for this token.
    text_id : ``int``, optional
        If your tokenizer returns integers instead of strings (e.g., because you're doing byte
        encoding, or some hash-based embedding), set this with the integer.  If this is set, we
        will bypass the vocabulary when indexing this token, regardless of whether ``text`` is also
        set.  You can `also` set ``text`` with the original text, if you want, so that you can
        still use a character-level representation in addition to a hash-based word embedding.

        The other fields on ``Token`` follow the fields on spacy's ``Token`` object; this is one we
        added, similar to spacy's ``lex_id``.
    """

    def __init__(self, text: str=None, idx: int=None, lemma: str=None, pos: str=None, tag: str=None, dep: str=None, ent_type: str=None, text_id: int=None) ->None:
        self.text = text
        self.idx = idx
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.ent_type_ = ent_type
        self.text_id = text_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


T = TypeVar('T')


_NO_DEFAULT = inspect.Parameter.empty


def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        return annotation


def takes_arg(obj, arg: str) ->bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f'object {obj} is not callable')
    return arg in signature.parameters


def create_kwargs(cls: Type[T], params: Params, **extras) ->Dict[str, Any]:
    """
    Given some class, a `Params` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the `extras` will just be used as is.
    For instance, you might provide an existing `Vocabulary` this way.
    """
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])
        default = param.default
        optional = default != _NO_DEFAULT
        if name in extras:
            kwargs[name] = extras[name]
        elif hasattr(annotation, 'from_params'):
            if name in params:
                subparams = params.pop(name)
                if takes_arg(annotation.from_params, 'extras'):
                    subextras = extras
                else:
                    subextras = {k: v for k, v in extras.items() if takes_arg(annotation.from_params, k)}
                if isinstance(subparams, str):
                    kwargs[name] = annotation.by_name(subparams)()
                else:
                    kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif not optional:
                raise ConfigurationError(f'expected key {name} for {cls.__name__}')
            else:
                kwargs[name] = default
        elif annotation == str:
            kwargs[name] = params.pop(name, default) if optional else params.pop(name)
        elif annotation == int:
            kwargs[name] = params.pop_int(name, default) if optional else params.pop_int(name)
        elif annotation == bool:
            kwargs[name] = params.pop_bool(name, default) if optional else params.pop_bool(name)
        elif annotation == float:
            kwargs[name] = params.pop_float(name, default) if optional else params.pop_float(name)
        elif origin in (Dict, dict) and len(args) == 2 and hasattr(args[-1], 'from_params'):
            value_cls = annotation.__args__[-1]
            value_dict = {}
            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)
            kwargs[name] = value_dict
        elif optional:
            kwargs[name] = params.pop(name, default)
        else:
            kwargs[name] = params.pop(name)
    params.assert_empty(cls.__name__)
    return kwargs


class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) ->T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses `FromParams`
        (or `Registrable`, which itself subclasses `FromParams`) gets this implementation for free.
        If you want your class to be instantiated from params in the "obvious" way -- pop off parameters
        and hand them to your constructor with the same names -- this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.
        """
        logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} and extras {extras}")
        if params is None:
            return None
        registered_subclasses = Registrable._registry.get(cls)
        if registered_subclasses is not None:
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice('type', choices=as_registrable.list_available(), default_to_first_choice=default_to_first_choice)
            subclass = registered_subclasses[choice]
            if not takes_arg(subclass.from_params, 'extras'):
                extras = {k: v for k, v in extras.items() if takes_arg(subclass.from_params, k)}
            return subclass.from_params(params=params, **extras)
        else:
            if cls.__init__ == object.__init__:
                kwargs: Dict[str, Any] = {}
            else:
                kwargs = create_kwargs(cls, params, **extras)
            return cls(**kwargs)


class Registrable(FromParams):
    """
    Any class that inherits from ``Registrable`` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    ``@BaseClass.register(name)``.

    After which you can call ``BaseClass.list_available()`` to get the keys for the
    registered subclasses, and ``BaseClass.by_name(name)`` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call ``from_params(params)`` on the returned subclass.

    You can specify a default by setting ``BaseClass.default_implementation``.
    If it is set, it will be the first element of ``list_available()``.

    Note that if you use this class to implement a new ``Registrable`` abstract class,
    you must ensure that all subclasses of the abstract class are loaded when the module is
    loaded, because the subclasses register themselves in their respective files. You can
    achieve this by having the abstract class and all subclasses in the __init__.py of the
    module in which they reside (as this causes any import of either the abstract class or
    a subclass to load all other subclasses and the abstract class).
    """
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = 'Cannot register %s as %s; name already in use for %s' % (name, cls.__name__, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) ->Type[T]:
        logger.info(f'instantiating registered subclass {name} of {cls}')
        if name not in Registrable._registry[cls]:
            raise ConfigurationError('%s is not a registered name for %s' % (name, cls.__name__))
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) ->List[str]:
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation
        if default is None:
            return keys
        elif default not in keys:
            message = 'Default implementation %s is not registered' % default
            raise ConfigurationError(message)
        else:
            return [default] + [k for k in keys if k != default]


class Tokenizer(Registrable):
    """
    A ``Tokenizer`` splits strings of text into tokens.  Typically, this either splits text into
    word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
    here, though you could imagine wanting to do other kinds of tokenization for structured or
    other inputs.

    As part of tokenization, concrete implementations of this API will also handle stemming,
    stopword filtering, adding start and end tokens, or other kinds of things you might want to do
    to your tokens.  See the parameters to, e.g., :class:`~.WordTokenizer`, or whichever tokenizer
    you want to use.

    If the base input to your model is words, you should use a :class:`~.WordTokenizer`, even if
    you also want to have a character-level encoder to get an additional vector for each word
    token.  Splitting word tokens into character arrays is handled separately, in the
    :class:`..token_representations.TokenRepresentation` class.
    """
    default_implementation = 'word'

    def batch_tokenize(self, texts: List[str]) ->List[List[Token]]:
        """
        Batches together tokenization of several texts, in case that is faster for particular
        tokenizers.
        """
        raise NotImplementedError

    def tokenize(self, text: str) ->List[Token]:
        """
        Actually implements splitting words into tokens.

        Returns
        -------
        tokens : ``List[Token]``
        """
        raise NotImplementedError


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


RnnStateStorage = Tuple[torch.Tensor, ...]


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError('Both the tensor and sequence lengths must be torch.Tensors.')
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = sequence_lengths.new_tensor(torch.arange(0, len(sequence_lengths)))
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


class _EncoderBase(torch.nn.Module):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/modules/encoder_base.py

    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`
    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool=False) ->None:
        super(_EncoderBase, self).__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence, Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor], RnnState]], inputs: torch.Tensor, mask: torch.Tensor, hidden_state: Optional[RnnState]=None):
        """
        Adopted from AllenNLP:
            https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/modules/encoder_base.py

        This function exists because Pytorch RNNs require that their inputs be sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a ``PackedSequence`` and some ``hidden_state``, which can either be a
        tuple of tensors or a tensor.
        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module, which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.
        Parameters
        ----------
        module : ``Callable[[PackedSequence, Optional[RnnState]],
                            Tuple[Union[PackedSequence, torch.Tensor], RnnState]]``, required.
            A function to run on the inputs. In most cases, this is a ``torch.nn.Module``.
        inputs : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length, embedding_size)`` representing
            the inputs to the Encoder.
        mask : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length)``, representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : ``Optional[RnnState]``, (default = None).
            A single tensor of shape (num_layers, batch_size, hidden_size) representing the
            state of an RNN with or a tuple of
            tensors of shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden state and memory
            state of an LSTM-like RNN.
        Returns
        -------
        module_output : ``Union[torch.Tensor, PackedSequence]``.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
            The batch size dimension will be equal to ``num_valid``, as sequences of zero
            length are clipped off before the module is called, as Pytorch cannot handle
            zero length sequences.
        final_states : ``Optional[RnnState]``
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : ``torch.LongTensor``
            A tensor of shape ``(batch_size,)``, describing the re-indexing required to transform
            the outputs back to their original batch order.
        """
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :], sorted_sequence_lengths[:num_valid].data.tolist(), batch_first=True)
        if not self.stateful:
            if hidden_state is None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous() for state in hidden_state]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)
        module_output, final_states = module(packed_sequence_input, initial_states)
        return module_output, final_states, restoration_indices

    def _get_initial_states(self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor) ->Optional[RnnState]:
        """
        Returns an initial state for use in an RNN. Additionally, this method handles
        the batch size changing across calls by mutating the state to append initial states
        for new elements in the batch. Finally, it also handles sorting the states
        with respect to the sequence lengths of elements in the batch and removing rows
        which are completely padded. Importantly, this `mutates` the state if the
        current batch size is larger than when it was previously called.
        Parameters
        ----------
        batch_size : ``int``, required.
            The batch size can change size across calls to stateful RNNs, so we need
            to know if we need to expand or shrink the states before returning them.
            Expanded states will be set to zero.
        num_valid : ``int``, required.
            The batch may contain completely padded sequences which get removed before
            the sequence is passed through the encoder. We also need to clip these off
            of the state too.
        sorting_indices ``torch.LongTensor``, required.
            Pytorch RNNs take sequences sorted by length. When we return the states to be
            used for a given call to ``module.forward``, we need the states to match up to
            the sorted sequences, so before returning them, we sort the states using the
            same indices used to sort the sequences.
        Returns
        -------
        This method has a complex return type because it has to deal with the first time it
        is called, when it has no state, and the fact that types of RNN have heterogeneous
        states.
        If it is the first time the module has been called, it returns ``None``, regardless
        of the type of the ``Module``.
        Otherwise, for LSTMs, it returns a tuple of ``torch.Tensors`` with shape
        ``(num_layers, num_valid, state_size)`` and ``(num_layers, num_valid, memory_size)``
        respectively, or for GRUs, it returns a single ``torch.Tensor`` of shape
        ``(num_layers, num_valid, state_size)``.
        """
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = state.new_zeros(state.size(0), num_states_to_concat, state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :]
        else:
            sorted_states = [state.index_select(1, sorting_indices) for state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :] for state in sorted_states)

    def _update_states(self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor) ->None:
        """
        After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detaches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.
        Parameters
        ----------
        final_states : ``RnnStateStorage``, required.
            The hidden states returned as output from the RNN.
        restoration_indices : ``torch.LongTensor``, required.
            The indices that invert the sorting used in ``sort_and_run_forward``
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        new_unsorted_states = [state.index_select(1, restoration_indices) for state in final_states]
        if self._states is None:
            self._states = tuple(state.data for state in new_unsorted_states)
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [(state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1) for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None


class Seq2VecEncoder(_EncoderBase):
    """
    A ``Seq2VecEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    single vector.  Input shape: ``(batch_size, sequence_length, input_dim)``; output shape:
    ``(batch_size, output_dim)``.

    We add two methods to the basic ``Module`` API: :func:`get_input_dim()` and :func:`get_output_dim()`.
    You might need this if you want to construct a ``Linear`` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """

    def get_input_dim(self) ->int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) ->int:
        """
        Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


class Metric:
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) ->Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) ->None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class BiLinear(nn.Module):
    """
    Bi-linear layer
    """

    def __init__(self, left_features, right_features, out_features, bias=True):
        """
        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features
        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """
        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]
        Returns:
        """
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], 'batch size of left and right inputs mis-match: (%s, %s)' % (left_size[:-1], right_size[:-1])
        batch = int(np.prod(left_size[:-1]))
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'in1_features=' + str(self.left_features) + ', in2_features=' + str(self.right_features) + ', out_features=' + str(self.out_features) + ')'


def _find_cycle(parents: List[int], length: int, current_nodes: List[bool]) ->Tuple[bool, List[int]]:
    """
    :return:
        has_cycle: whether the graph has at least a cycle.
        cycle: a list of nodes which form a cycle in the graph.
    """
    added = [(False) for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        if added[i] or not current_nodes[i]:
            continue
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)
        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break
    return has_cycle, list(cycle)


def chu_liu_edmonds(length: int, score_matrix: numpy.ndarray, current_nodes: List[bool], final_edges: Dict[int, int], old_input: numpy.ndarray, old_output: numpy.ndarray, representatives: List[Set[int]]):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.
    Note that this function operates in place, so variables
    will be modified.
    Parameters
    ----------
    length : ``int``, required.
        The number of nodes.
    score_matrix : ``numpy.ndarray``, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : ``List[bool]``, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges: ``Dict[int, int]``, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input: ``numpy.ndarray``, required.
        a map from an edge to its head node.
        Key: The edge is a tuple, and elements in a tuple
        could be a node or a representative of a cycle.
    old_output: ``numpy.ndarray``, required.
    representatives : ``List[Set[int]]``, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.
    Returns
    -------
    Nothing - all variables are modified in place.
    """
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue
                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue
            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return
    cycle_weight = 0.0
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue
        in_edge_weight = float('-inf')
        in_edge = -1
        out_edge_weight = float('-inf')
        out_edge = -1
        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle
            score = cycle_weight + score_matrix[node, node_in_cycle] - score_matrix[parents[node_in_cycle], node_in_cycle]
            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle
        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]
        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            current_nodes[node_in_cycle] = False
        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)
    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break
    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def decode_mst(energy: numpy.ndarray, length: int, has_labels: bool=True) ->Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.
    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arboresences on graphs.
    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ConfigurationError('The dimension of the energy array is not equal to 3.')
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError('The dimension of the energy array is not equal to 2.')
    input_shape = energy.shape
    max_length = input_shape[-1]
    if has_labels:
        energy = energy[:, :length, :length]
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    original_score_matrix = energy
    score_matrix = numpy.array(original_score_matrix, copy=True)
    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [(True) for _ in range(length)]
    representatives: List[Set[int]] = []
    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})
        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2
            old_input[node2, node1] = node2
            old_output[node2, node1] = node1
    final_edges: Dict[int, int] = {}
    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None
    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]
    return heads, head_type


def _validate(final_edges, length, original_score_matrix, coreference):
    modified = 0
    current_nodes = [(True) for _ in range(length)]
    group_by_precedent = {}
    for node, precedent in enumerate(coreference):
        if precedent not in group_by_precedent:
            group_by_precedent[precedent] = []
        group_by_precedent[precedent].append(node)
    for group in group_by_precedent.values():
        if len(group) == 1:
            continue
        conflicts_by_parent = {}
        for child in group:
            parent = final_edges[child]
            if parent not in conflicts_by_parent:
                conflicts_by_parent[parent] = []
            conflicts_by_parent[parent].append(child)
        reserved_parents = set(conflicts_by_parent.keys())
        for parent, conflicts in conflicts_by_parent.items():
            if len(conflicts) == 1:
                continue
            winner = max(conflicts, key=lambda _child: original_score_matrix[parent, _child])
            for child in conflicts:
                if child == winner:
                    continue
                parent_scores = original_score_matrix[:, child]
                for _parent in numpy.argsort(parent_scores)[::-1]:
                    if _parent == parent or _parent in reserved_parents:
                        continue
                    parents = final_edges.copy()
                    parents[child] = _parent
                    has_cycle, _ = _find_cycle(parents, length, current_nodes)
                    if has_cycle:
                        continue
                    reserved_parents.add(_parent)
                    final_edges[child] = _parent
                    modified += 1
                    break
    return modified


def adapted_chu_liu_edmonds(length: int, score_matrix: numpy.ndarray, coreference: List[int], current_nodes: List[bool], final_edges: Dict[int, int], old_input: numpy.ndarray, old_output: numpy.ndarray, representatives: List[Set[int]]):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.
    Note that this function operates in place, so variables
    will be modified.
    Parameters
    ----------
    length : ``int``, required.
        The number of nodes.
    score_matrix : ``numpy.ndarray``, required.
        The score matrix representing the scores for pairs
        of nodes.
    coreference: ``List[int]``, required.
        A list which maps a node to its first precedent.
    current_nodes : ``List[bool]``, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges: ``Dict[int, int]``, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input: ``numpy.ndarray``, required.
        a map from an edge to its head node.
        Key: The edge is a tuple, and elements in a tuple
        could be a node or a representative of a cycle.
    old_output: ``numpy.ndarray``, required.
    representatives : ``List[Set[int]]``, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.
    Returns
    -------
    Nothing - all variables are modified in place.
    """
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue
                _parent = old_input[node1, node2]
                _child = old_output[node1, node2]
                if coreference[_parent] == coreference[_child]:
                    continue
                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue
            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return
    cycle_weight = 0.0
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue
        in_edge_weight = float('-inf')
        in_edge = -1
        out_edge_weight = float('-inf')
        out_edge = -1
        for node_in_cycle in cycle:
            _parent = old_input[node_in_cycle, node]
            _child = old_output[node_in_cycle, node]
            if coreference[_parent] != coreference[_child]:
                if score_matrix[node_in_cycle, node] > in_edge_weight:
                    in_edge_weight = score_matrix[node_in_cycle, node]
                    in_edge = node_in_cycle
            _parent = old_input[node, node_in_cycle]
            _child = old_output[node, node_in_cycle]
            if coreference[_parent] != coreference[_child]:
                score = cycle_weight + score_matrix[node, node_in_cycle] - score_matrix[parents[node_in_cycle], node_in_cycle]
                if score > out_edge_weight:
                    out_edge_weight = score
                    out_edge = node_in_cycle
        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]
        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            current_nodes[node_in_cycle] = False
        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)
    adapted_chu_liu_edmonds(length, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break
    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def decode_mst_with_coreference(energy: numpy.ndarray, coreference: List[int], length: int, has_labels: bool=True) ->Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.
    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arboresences on graphs.
    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    coreference: ``List[int]``, required.
        A list which maps a node to its first precedent.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ConfigurationError('The dimension of the energy array is not equal to 3.')
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError('The dimension of the energy array is not equal to 2.')
    input_shape = energy.shape
    max_length = input_shape[-1]
    if has_labels:
        energy = energy[:, :length, :length]
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    original_score_matrix = energy
    score_matrix = numpy.array(original_score_matrix, copy=True)
    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [(True) for _ in range(length)]
    representatives: List[Set[int]] = []
    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})
        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2
            old_input[node2, node1] = node2
            old_output[node2, node1] = node1
    final_edges: Dict[int, int] = {}
    adapted_chu_liu_edmonds(length, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)
    _validate(final_edges, length, original_score_matrix, coreference)
    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None
    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]
    return heads, head_type


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int=-1) ->torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


class DeepBiaffineGraphDecoder(torch.nn.Module):

    def __init__(self, decode_algorithm, head_sentinel, edge_node_h_linear, edge_node_m_linear, edge_label_h_linear, edge_label_m_linear, encode_dropout, biaffine_attention, edge_label_bilinear):
        super(DeepBiaffineGraphDecoder, self).__init__()
        self.decode_algorithm = decode_algorithm
        self.head_sentinel = head_sentinel
        self.edge_node_h_linear = edge_node_h_linear
        self.edge_node_m_linear = edge_node_m_linear
        self.edge_label_h_linear = edge_label_h_linear
        self.edge_label_m_linear = edge_label_m_linear
        self.encode_dropout = encode_dropout
        self.biaffine_attention = biaffine_attention
        self.edge_label_bilinear = edge_label_bilinear
        self.metrics = AttachmentScores()
        self.minus_inf = -100000000.0

    def forward(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        num_nodes = mask.sum().item()
        memory_bank, edge_heads, edge_labels, corefs, mask = self._add_head_sentinel(memory_bank, edge_heads, edge_labels, corefs, mask)
        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.encode(memory_bank)
        edge_node_scores = self._get_edge_node_scores(edge_node_h, edge_node_m, mask)
        edge_node_nll, edge_label_nll = self.get_loss(edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask)
        pred_edge_heads, pred_edge_labels = self.decode(edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        self.metrics(pred_edge_heads, pred_edge_labels, edge_heads[:, 1:], edge_labels[:, 1:], mask[:, 1:], edge_node_nll.item(), edge_label_nll.item())
        return dict(edge_heads=pred_edge_heads, edge_labels=pred_edge_labels, loss=(edge_node_nll + edge_label_nll) / num_nodes, total_loss=edge_node_nll + edge_label_nll, num_nodes=torch.tensor(float(num_nodes)).type_as(memory_bank))

    def encode(self, memory_bank):
        """
        Map contextual representation into specific space (w/ lower dimensionality).

        :param input: [batch, length, hidden_size]
        :return:
            edge_node: a tuple of (head, modifier) hidden state with size [batch, length, edge_hidden_size]
            edge_label: a tuple of (head, modifier) hidden state with size [batch, length, label_hidden_size]
        """
        edge_node_h = torch.nn.functional.elu(self.edge_node_h_linear(memory_bank))
        edge_node_m = torch.nn.functional.elu(self.edge_node_m_linear(memory_bank))
        edge_label_h = torch.nn.functional.elu(self.edge_label_h_linear(memory_bank))
        edge_label_m = torch.nn.functional.elu(self.edge_label_m_linear(memory_bank))
        edge_node = torch.cat([edge_node_h, edge_node_m], dim=1)
        edge_label = torch.cat([edge_label_h, edge_label_m], dim=1)
        edge_node = self.encode_dropout(edge_node.transpose(1, 2)).transpose(1, 2)
        edge_label = self.encode_dropout(edge_label.transpose(1, 2)).transpose(1, 2)
        edge_node_h, edge_node_m = edge_node.chunk(2, 1)
        edge_label_h, edge_label_m = edge_label.chunk(2, 1)
        return (edge_node_h, edge_node_m), (edge_label_h, edge_label_m)

    def get_loss(self, edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask):
        """
        :param edge_label_h: [batch, length, hidden_size]
        :param edge_label_m: [batch, length, hidden_size]
        :param edge_node_scores:  [batch, length, length]
        :param edge_heads:  [batch, length]
        :param edge_labels:  [batch, length]
        :param mask: [batch, length]
        :return:  [batch, length - 1]
        """
        batch_size, max_len, _ = edge_node_scores.size()
        edge_node_log_likelihood = masked_log_softmax(edge_node_scores, mask.unsqueeze(2) + mask.unsqueeze(1), dim=1)
        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        edge_label_log_likelihood = torch.nn.functional.log_softmax(edge_label_scores, dim=2)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        modifier_index = torch.arange(0, max_len).view(1, max_len).expand(batch_size, max_len).type_as(edge_heads)
        _edge_node_log_likelihood = edge_node_log_likelihood[batch_index, edge_heads.data, modifier_index]
        _edge_label_log_likelihood = edge_label_log_likelihood[batch_index, modifier_index, edge_labels.data]
        gold_edge_node_nll = -_edge_node_log_likelihood[:, 1:].sum()
        gold_edge_label_nll = -_edge_label_log_likelihood[:, 1:].sum()
        return gold_edge_node_nll, gold_edge_label_nll

    def decode(self, edge_label_h, edge_label_m, edge_node_scores, corefs, mask):
        if self.decode_algorithm == 'mst':
            return self.mst_decode(edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        else:
            return self.greedy_decode(edge_label_h, edge_label_m, edge_node_scores, mask)

    def greedy_decode(self, edge_label_h, edge_label_m, edge_node_scores, mask):
        edge_node_scores = edge_node_scores.data
        max_len = edge_node_scores.size(1)
        edge_node_scores = edge_node_scores + torch.diag(edge_node_scores.new(max_len).fill_(-np.inf))
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_node_scores = edge_node_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        _, edge_heads = edge_node_scores.max(dim=1)
        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        _, edge_labels = edge_label_scores.max(dim=2)
        return edge_heads[:, 1:], edge_labels[:, 1:]

    def mst_decode(self, edge_label_h, edge_label_m, edge_node_scores, corefs, mask):
        batch_size, max_length, edge_label_hidden_size = edge_label_h.size()
        lengths = mask.data.sum(dim=1).long().cpu().numpy()
        expanded_shape = [batch_size, max_length, max_length, edge_label_hidden_size]
        edge_label_h = edge_label_h.unsqueeze(2).expand(*expanded_shape).contiguous()
        edge_label_m = edge_label_m.unsqueeze(1).expand(*expanded_shape).contiguous()
        edge_label_scores = self.edge_label_bilinear(edge_label_h, edge_label_m)
        edge_label_scores = torch.nn.functional.log_softmax(edge_label_scores, dim=3).permute(0, 3, 1, 2)
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_node_scores = edge_node_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        edge_node_scores = torch.nn.functional.log_softmax(edge_node_scores, dim=1)
        batch_energy = torch.exp(edge_node_scores.unsqueeze(1) + edge_label_scores)
        edge_heads, edge_labels = self._run_mst_decoding(batch_energy, lengths, corefs)
        return edge_heads[:, 1:], edge_labels[:, 1:]

    @staticmethod
    def _run_mst_decoding(batch_energy, lengths, corefs=None):
        edge_heads = []
        edge_labels = []
        for i, (energy, length) in enumerate(zip(batch_energy.detach().cpu(), lengths)):
            scores, label_ids = energy.max(dim=0)
            scores[0, :] = 0
            if corefs is not None:
                coref = corefs[i].detach().cpu().tolist()[:length]
                instance_heads, _ = decode_mst_with_coreference(scores.numpy(), coref, length, has_labels=False)
            else:
                instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)
            instance_head_labels = []
            for child, parent in enumerate(instance_heads):
                instance_head_labels.append(label_ids[parent, child].item())
            instance_heads[0] = 0
            instance_head_labels[0] = 0
            edge_heads.append(instance_heads)
            edge_labels.append(instance_head_labels)
        return torch.from_numpy(np.stack(edge_heads)), torch.from_numpy(np.stack(edge_labels))

    @classmethod
    def from_params(cls, vocab, params):
        decode_algorithm = params['decode_algorithm']
        input_size = params['input_size']
        edge_node_hidden_size = params['edge_node_hidden_size']
        edge_label_hidden_size = params['edge_label_hidden_size']
        dropout = params['dropout']
        head_sentinel = torch.nn.Parameter(torch.randn([1, 1, input_size]))
        edge_node_h_linear = torch.nn.Linear(input_size, edge_node_hidden_size)
        edge_node_m_linear = torch.nn.Linear(input_size, edge_node_hidden_size)
        edge_label_h_linear = torch.nn.Linear(input_size, edge_label_hidden_size)
        edge_label_m_linear = torch.nn.Linear(input_size, edge_label_hidden_size)
        encode_dropout = torch.nn.Dropout2d(p=dropout)
        biaffine_attention = BiaffineAttention(edge_node_hidden_size, edge_node_hidden_size)
        num_labels = vocab.get_vocab_size('head_tags')
        edge_label_bilinear = BiLinear(edge_label_hidden_size, edge_label_hidden_size, num_labels)
        return cls(decode_algorithm=decode_algorithm, head_sentinel=head_sentinel, edge_node_h_linear=edge_node_h_linear, edge_node_m_linear=edge_node_m_linear, edge_label_h_linear=edge_label_h_linear, edge_label_m_linear=edge_label_m_linear, encode_dropout=encode_dropout, biaffine_attention=biaffine_attention, edge_label_bilinear=edge_label_bilinear)

    def _add_head_sentinel(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        """
        Add a dummy ROOT at the beginning of each node sequence.
        :param memory_bank: [batch, length, hidden_size]
        :param edge_head: None or [batch, length]
        :param edge_labels: None or [batch, length]
        :param corefs: None or [batch, length]
        :param mask: [batch, length]
        """
        batch_size, _, hidden_size = memory_bank.size()
        head_sentinel = self.head_sentinel.expand([batch_size, 1, hidden_size])
        memory_bank = torch.cat([head_sentinel, memory_bank], 1)
        if edge_heads is not None:
            edge_heads = torch.cat([edge_heads.new_zeros(batch_size, 1), edge_heads], 1)
        if edge_labels is not None:
            edge_labels = torch.cat([edge_labels.new_zeros(batch_size, 1), edge_labels], 1)
        if corefs is not None:
            corefs = torch.cat([corefs.new_zeros(batch_size, 1), corefs], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        return memory_bank, edge_heads, edge_labels, corefs, mask

    def _get_edge_node_scores(self, edge_node_h, edge_node_m, mask):
        edge_node_scores = self.biaffine_attention(edge_node_h, edge_node_m, mask_d=mask, mask_e=mask).squeeze(1)
        return edge_node_scores

    def _get_edge_label_scores(self, edge_label_h, edge_label_m, edge_heads):
        """
        Compute the edge label scores.
        :param edge_label_h: [batch, length, edge_label_hidden_size]
        :param edge_label_m: [batch, length, edge_label_hidden_size]
        :param heads: [batch, length] -- element at [i, j] means the head index of node_j at batch_i.
        :return: [batch, length, num_labels]
        """
        batch_size = edge_label_h.size(0)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads.data).long()
        edge_label_h = edge_label_h[batch_index, edge_heads.data].contiguous()
        edge_label_m = edge_label_m.contiguous()
        edge_label_scores = self.edge_label_bilinear(edge_label_h, edge_label_m)
        return edge_label_scores


class DotProductAttention(torch.nn.Module):

    def __init__(self, decoder_hidden_size, encoder_hidden_size, share_linear=True):
        super(DotProductAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.linear_layer = torch.nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)
        self.share_linear = share_linear

    def forward(self, decoder_input, encoder_input):
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        decoder_input = self.linear_layer(decoder_input)
        if self.share_linear:
            encoder_input = self.linear_layer(encoder_input)
        encoder_input = encoder_input.transpose(1, 2)
        return torch.bmm(decoder_input, encoder_input)


END_SYMBOL = '@end@'


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError('No dimension to distribute: ' + str(input_size))
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))
        reshaped_outputs = self._module(*reshaped_inputs)
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)
        return outputs


class TokenEmbedder(torch.nn.Module):
    """
    A ``TokenEmbedder`` is a ``Module`` that takes as input a tensor with integer ids that have
    been output from a :class:`~allennlp.data.TokenIndexer` and outputs a vector per token in the
    input.  The input typically has shape ``(batch_size, num_tokens)`` or ``(batch_size,
    num_tokens, num_characters)``, and the output is of shape ``(batch_size, num_tokens,
    output_dim)``.  The simplest ``TokenEmbedder`` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.

    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  This lets us
    more easily compute output dimensions for the :class:`~allennlp.modules.TextFieldEmbedder`,
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """
    default_implementation = 'embedding'

    def get_output_dim(self) ->int:
        """
        Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


def _read_embeddings_from_hdf5(embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str='tokens', amr: bool=False) ->torch.FloatTensor:
    """
    Reads from a hdf5 formatted file. The embedding matrix is assumed to
    be keyed by 'embedding' and of size ``(num_tokens, embedding_dim)``.
    """
    with h5py.File(embeddings_filename, 'r') as fin:
        embeddings = fin['embedding'][...]
    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise ConfigurationError('Read shape {0} embeddings from the file, but expected {1}'.format(list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]))
    return torch.FloatTensor(embeddings)


def http_get(url: str, temp_file: IO) ->None:
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = Tqdm.tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise FileNotFoundError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url: str) ->Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


@s3_request
def s3_etag(url: str) ->Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = boto3.resource('s3')
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url: str, temp_file: IO) ->None:
    """Pull a file directly from S3."""
    s3_resource = boto3.resource('s3')
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def url_to_filename(url: str, etag: str=None) ->str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    return filename


def get_from_cache(url: str, cache_dir: str=None) ->str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('s3://'):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError('HEAD request failed for url {} with status code {}'.format(url, response.status_code))
        etag = response.headers.get('ETag')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info('%s not found in cache, downloading to %s', url, temp_file.name)
            if url.startswith('s3://'):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            logger.info('copying %s to cache at %s', temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info('creating metadata file for %s', cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)
            logger.info('removing temp file %s', temp_file.name)
    return cache_path


def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str]=None) ->str:
    if path_inside_archive:
        return '({})#{}'.format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


def get_file_extension(path: str, dot=True, lower: bool=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


class EmbeddingsFileURI(NamedTuple):
    main_file_uri: str
    path_inside_archive: Optional[str] = None


def parse_embeddings_file_uri(uri: str) ->'EmbeddingsFileURI':
    match = re.fullmatch('\\((.*)\\)#(.*)', uri)
    if match:
        fields = cast(Tuple[str, str], match.groups())
        return EmbeddingsFileURI(*fields)
    else:
        return EmbeddingsFileURI(uri, None)


class EmbeddingsTextFile(Iterator[str]):
    """
    Utility class for opening embeddings text files. Handles various compression formats,
    as well as context management.

    Parameters
    ----------
    file_uri: str
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.
        * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
          is contained in a multi-file archive.

    encoding: str
    cache_dir: str
    """
    DEFAULT_ENCODING = 'utf-8'

    def __init__(self, file_uri: str, encoding: str=DEFAULT_ENCODING, cache_dir: str=None) ->None:
        self.uri = file_uri
        self._encoding = encoding
        self._cache_dir = cache_dir
        self._archive_handle: Any = None
        main_file_uri, path_inside_archive = parse_embeddings_file_uri(file_uri)
        main_file_local_path = cached_path(main_file_uri, cache_dir=cache_dir)
        if zipfile.is_zipfile(main_file_local_path):
            self._open_inside_zip(main_file_uri, path_inside_archive)
        elif tarfile.is_tarfile(main_file_local_path):
            self._open_inside_tar(main_file_uri, path_inside_archive)
        else:
            if path_inside_archive:
                raise ValueError('Unsupported archive format: %s' + main_file_uri)
            extension = get_file_extension(main_file_uri)
            package = {'.txt': io, '.vec': io, '.gz': gzip, '.bz2': bz2, '.lzma': lzma}.get(extension, None)
            if package is None:
                logger.warning('The embeddings file has an unknown file extension "%s". We will assume the file is an (uncompressed) text file', extension)
                package = io
            self._handle = package.open(main_file_local_path, 'rt', encoding=encoding)
        first_line = next(self._handle)
        self.num_tokens = EmbeddingsTextFile._get_num_tokens_from_first_line(first_line)
        if self.num_tokens:
            self._iterator = self._handle
        else:
            self._iterator = itertools.chain([first_line], self._handle)

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str]=None) ->None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = zipfile.ZipFile(cached_archive_path, 'r')
        if member_path is None:
            members_list = archive.namelist()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member_file = archive.open(member_path, 'r')
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str]=None) ->None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = tarfile.open(cached_archive_path, 'r')
        if member_path is None:
            members_list = archive.getnames()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member = archive.getmember(member_path)
        member_file = cast(IO[bytes], archive.extractfile(member))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def read(self) ->str:
        return ''.join(self._iterator)

    def readline(self) ->str:
        return next(self._iterator)

    def close(self) ->None:
        self._handle.close()
        if self._archive_handle:
            self._archive_handle.close()

    def __enter__(self) ->'EmbeddingsTextFile':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) ->None:
        self.close()

    def __iter__(self) ->'EmbeddingsTextFile':
        return self

    def __next__(self) ->str:
        return next(self._iterator)

    def __len__(self) ->Optional[int]:
        """ Hack for tqdm: no need for explicitly passing ``total=file.num_tokens`` """
        if self.num_tokens:
            return self.num_tokens
        raise AttributeError('an object of type EmbeddingsTextFile has "len()" only if the underlying text file declares the number of tokens (i.e. the number of lines following)in the first line. That is not the case of this particular instance.')

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) ->str:
        if len(members_list) > 1:
            raise ValueError('The archive %s contains multiple files, so you must select one of the files inside providing a uri of the type: %s' % (archive_path, format_embeddings_file_uri('path_or_url_to_archive', 'path_inside_archive')))
        return members_list[0]

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) ->Optional[int]:
        """ This function takes in input a string and if it contains 1 or 2 integers, it assumes the
        largest one it the number of tokens. Returns None if the line doesn't match that pattern. """
        fields = line.split(' ')
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None
            else:
                num_tokens = max(int_fields)
                logger.info('Recognized a header line in the embedding file with number of tokens: %d', num_tokens)
                return num_tokens
        return None


def _read_embeddings_from_text_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str='tokens', amr: bool=False) ->torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.

    The remainder of the docstring is identical to ``_read_pretrained_embeddings_file``.
    """
    tokens_to_keep = set()
    for token in vocab.get_token_to_index_vocabulary(namespace):
        if amr:
            token = re.sub('-\\d\\d$', '', token)
        tokens_to_keep.add(token)
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}
    logger.info('Reading pretrained embeddings from file')
    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(' ', 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != embedding_dim:
                    logger.warning('Found line with wrong number of dimensions (expected: %d; actual: %d): %s', embedding_dim, len(fields) - 1, line)
                    continue
                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector
    if not embeddings:
        raise ConfigurationError("No embeddings of correct dimension found; you probably misspecified your embedding_dim parameter, or didn't pre-populate your Vocabulary")
    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    logger.info('Initializing pre-trained embedding layer')
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_std)
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            if amr:
                normalized_token = re.sub('-\\d\\d$', '', token)
                if normalized_token in embeddings:
                    embedding_matrix[i] = torch.FloatTensor(embeddings[normalized_token])
                    num_tokens_found += 1
            logger.debug('Token %s was not found in the embedding file. Initialising randomly.', token)
    logger.info('Pretrained embeddings were found for %d out of %d tokens', num_tokens_found, vocab_size)
    return embedding_matrix


def _read_pretrained_embeddings_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str='tokens', amr: bool=False) ->torch.FloatTensor:
    """
    Returns and embedding matrix for the given vocabulary using the pretrained embeddings
    contained in the given file. Embeddings for tokens not found in the pretrained embedding file
    are randomly initialized using a normal distribution with mean and standard deviation equal to
    those of the pretrained embeddings.

    We support two file formats:

        * text format - utf-8 encoded text file with space separated fields: [word] [dim 1] [dim 2] ...
          The text file can eventually be compressed, and even resides in an archive with multiple files.
          If the file resides in an archive with other files, then ``embeddings_filename`` must
          be a URI "(archive_uri)#file_path_inside_the_archive"

        * hdf5 format - hdf5 file containing an embedding matrix in the form of a torch.Tensor.

    If the filename ends with '.hdf5' or '.h5' then we load from hdf5, otherwise we assume
    text format.

    Parameters
    ----------
    file_uri : str, required.
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.

        * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
          is contained in a multi-file archive.

    vocab : Vocabulary, required.
        A Vocabulary object.
    namespace : str, (optional, default=tokens)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : bool, (optional, default=True)
        Whether or not the embedding parameters should be optimized.

    Returns
    -------
    A weight matrix with embeddings initialized from the read file.  The matrix has shape
    ``(vocab.get_vocab_size(namespace), embedding_dim)``, where the indices of words appearing in
    the pretrained embedding file are initialized to the pretrained embedding value.
    """
    file_ext = get_file_extension(file_uri)
    if file_ext in ['.h5', '.hdf5']:
        return _read_embeddings_from_hdf5(file_uri, embedding_dim, vocab, namespace, amr)
    return _read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace, amr)


class MLPAttention(torch.nn.Module):

    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention_hidden_size, coverage=False, use_concat=False):
        super(MLPAttention, self).__init__()
        self.hidden_size = attention_hidden_size
        self.query_linear = torch.nn.Linear(decoder_hidden_size, self.hidden_size, bias=True)
        self.context_linear = torch.nn.Linear(encoder_hidden_size, self.hidden_size, bias=False)
        self.output_linear = torch.nn.Linear(self.hidden_size, 1, bias=False)
        if coverage:
            self.coverage_linear = torch.nn.Linear(1, self.hidden_size, bias=False)
        self.use_concat = use_concat
        if self.use_concat:
            self.concat_linear = torch.nn.Linear(decoder_hidden_size, self.hidden_size, bias=False)

    def forward(self, decoder_input, encoder_input, coverage=None):
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :param coverage: [batch, encoder_seq_length]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        batch_size, decoder_seq_length, decoder_hidden_size = decoder_input.size()
        batch_size, encoder_seq_length, encoder_hidden_size = encoder_input.size()
        decoder_features = self.query_linear(decoder_input)
        decoder_features = decoder_features.unsqueeze(2).expand(batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
        encoder_features = self.context_linear(encoder_input)
        encoder_features = encoder_features.unsqueeze(1).expand(batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
        attn_features = decoder_features + encoder_features
        if coverage is not None:
            coverage_features = self.coverage_linear(coverage.view(batch_size, 1, encoder_seq_length, 1)).expand(batch_size, decoder_seq_length, encoder_seq_length, self.hidden_size)
            attn_features = attn_features + coverage_features
        if self.use_concat:
            concat_input = decoder_input.unsqueeze(2).expand(batch_size, decoder_seq_length, encoder_seq_length, decoder_hidden_size) * encoder_input.unsqueeze(1).expand(batch_size, decoder_seq_length, encoder_seq_length, encoder_hidden_size)
            concat_features = self.concat_linear(concat_input)
            attn_features = attn_features + concat_features
        e = torch.tanh(attn_features)
        scores = self.output_linear(e).squeeze(3)
        return scores


class GlobalAttention(torch.nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """

    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention):
        super(GlobalAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.attention = attention
        self.output_layer = torch.nn.Linear(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size, bias=isinstance(attention, MLPAttention))

    def forward(self, source, memory_bank, mask=None, coverage=None):
        """
        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          mask (`LongTensor`): the source context mask `[batch, length]`
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        batch_, target_l, dim_ = source.size()
        one_step = False
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        if isinstance(self.attention, MLPAttention) and coverage is not None:
            align = self.attention(source, memory_bank, coverage)
        elif isinstance(self.attention, BiaffineAttention):
            align = self.attention(source, memory_bank).squeeze(1)
        else:
            align = self.attention(source, memory_bank)
        if mask is not None:
            mask = mask.byte().unsqueeze(1)
            align.masked_fill_(1 - mask, -float('inf'))
        align_vectors = F.softmax(align, 2)
        c = torch.bmm(align_vectors, memory_bank)
        concat_c = torch.cat([c, source], 2).view(batch_ * target_l, -1)
        attn_h = self.output_layer(concat_c).view(batch_, target_l, -1)
        attn_h = torch.tanh(attn_h)
        if coverage is not None:
            coverage = coverage + align_vectors
        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        return attn_h, align_vectors, coverage


class RNNDecoderBase(torch.nn.Module):

    def __init__(self, rnn_cell, dropout):
        super(RNNDecoderBase, self).__init__()
        self.rnn_cell = rnn_cell
        self.dropout = dropout

    def forward(self, *input):
        raise NotImplementedError


class InputFeedRNNDecoder(RNNDecoderBase):

    def __init__(self, rnn_cell, dropout, attention_layer, source_copy_attention_layer=None, coref_attention_layer=None, use_coverage=False):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer
        self.source_copy_attention_layer = source_copy_attention_layer
        self.coref_attention_layer = coref_attention_layer
        self.use_coverage = use_coverage

    def forward(self, inputs, memory_bank, mask, hidden_state, input_feed=None, target_copy_hidden_states=None, coverage=None):
        """

        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: None or [batch_size, 1, hidden_size]
        :param target_copy_hidden_states: None or [batch_size, seq_length, hidden_size]
        :param coverage: None or [batch_size, 1, encode_seq_length]
        :return:
        """
        batch_size, sequence_length, _ = inputs.size()
        one_step_length = [1] * batch_size
        source_copy_attentions = []
        target_copy_attentions = []
        coverage_records = []
        decoder_hidden_states = []
        rnn_hidden_states = []
        if input_feed is None:
            input_feed = inputs.new_zeros(batch_size, 1, self.rnn_cell.hidden_size)
        if target_copy_hidden_states is None:
            target_copy_hidden_states = []
        if self.use_coverage and coverage is None:
            coverage = inputs.new_zeros(batch_size, 1, memory_bank.size(1))
        for step_i, input in enumerate(inputs.split(1, dim=1)):
            _input = torch.cat([input, input_feed], 2)
            packed_input = pack_padded_sequence(_input, one_step_length, batch_first=True)
            packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            rnn_hidden_states.append(output)
            coverage_records.append(coverage)
            output, std_attention, coverage = self.attention_layer(output, memory_bank, mask, coverage)
            output = self.dropout(output)
            input_feed = output
            if self.source_copy_attention_layer is not None:
                _, source_copy_attention = self.source_copy_attention_layer(output, memory_bank, mask)
                source_copy_attentions.append(source_copy_attention)
            else:
                source_copy_attentions.append(std_attention)
            if self.coref_attention_layer is not None:
                if len(target_copy_hidden_states) == 0:
                    target_copy_attention = inputs.new_zeros(batch_size, 1, sequence_length)
                else:
                    target_copy_memory = torch.cat(target_copy_hidden_states, 1)
                    if sequence_length == 1:
                        _, target_copy_attention, _ = self.coref_attention_layer(output, target_copy_memory)
                    else:
                        _, target_copy_attention, _ = self.coref_attention_layer(output, target_copy_memory)
                        target_copy_attention = torch.nn.functional.pad(target_copy_attention, (0, sequence_length - step_i), 'constant', 0)
                target_copy_attentions.append(target_copy_attention)
            target_copy_hidden_states.append(output)
            decoder_hidden_states.append(output)
        decoder_hidden_states = torch.cat(decoder_hidden_states, 1)
        rnn_hidden_states = torch.cat(rnn_hidden_states, 1)
        source_copy_attentions = torch.cat(source_copy_attentions, 1)
        if len(target_copy_attentions):
            target_copy_attentions = torch.cat(target_copy_attentions, 1)
        else:
            target_copy_attentions = None
        if self.use_coverage:
            coverage_records = torch.cat(coverage_records, 1)
        else:
            coverage_records = None
        return dict(decoder_hidden_states=decoder_hidden_states, rnn_hidden_states=rnn_hidden_states, source_copy_attentions=source_copy_attentions, target_copy_attentions=target_copy_attentions, coverage_records=coverage_records, last_hidden_state=hidden_state, input_feed=input_feed, coverage=coverage)


class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


class PointerGenerator(torch.nn.Module):

    def __init__(self, input_size, switch_input_size, vocab_size, vocab_pad_idx, force_copy):
        super(PointerGenerator, self).__init__()
        self.linear = torch.nn.Linear(input_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear_pointer = torch.nn.Linear(switch_input_size, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.metrics = Seq2SeqMetrics()
        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_pad_idx
        self.force_copy = force_copy
        self.eps = 1e-20

    def forward(self, hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps, invalid_indexes=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param hiddens: decoder outputs, [batch_size, num_target_nodes, hidden_size]
        :param source_attentions: attention of each source node,
            [batch_size, num_target_nodes, num_source_nodes]
        :param source_attention_maps: a sparse indicator matrix
            mapping each source node to its index in the dynamic vocabulary.
            [batch_size, num_source_nodes, dynamic_vocab_size]
        :param target_attentions: attention of each target node,
            [batch_size, num_target_nodes, num_target_nodes]
        :param target_attention_maps: a sparse indicator matrix
            mapping each target node to its index in the dynamic vocabulary.
            [batch_size, num_target_nodes, dynamic_vocab_size]
        :param invalid_indexes: indexes which are not considered in prediction.
        """
        batch_size, num_target_nodes, _ = hiddens.size()
        source_dynamic_vocab_size = source_attention_maps.size(2)
        target_dynamic_vocab_size = target_attention_maps.size(2)
        hiddens = hiddens.view(batch_size * num_target_nodes, -1)
        p = torch.nn.functional.softmax(self.linear_pointer(hiddens), dim=1)
        p_copy_source = p[:, 0].view(batch_size, num_target_nodes, 1)
        p_copy_target = p[:, 1].view(batch_size, num_target_nodes, 1)
        p_generate = p[:, 2].view(batch_size, num_target_nodes, 1)
        scores = self.linear(hiddens)
        scores[:, self.vocab_pad_idx] = -float('inf')
        scores = scores.view(batch_size, num_target_nodes, -1)
        vocab_probs = self.softmax(scores)
        scaled_vocab_probs = torch.mul(vocab_probs, p_generate.expand_as(vocab_probs))
        scaled_source_attentions = torch.mul(source_attentions, p_copy_source.expand_as(source_attentions))
        scaled_copy_source_probs = torch.bmm(scaled_source_attentions, source_attention_maps.float())
        scaled_target_attentions = torch.mul(target_attentions, p_copy_target.expand_as(target_attentions))
        scaled_copy_target_probs = torch.bmm(scaled_target_attentions, target_attention_maps.float())
        if invalid_indexes:
            if invalid_indexes.get('vocab', None) is not None:
                vocab_invalid_indexes = invalid_indexes['vocab']
                for i, indexes in enumerate(vocab_invalid_indexes):
                    for index in indexes:
                        scaled_vocab_probs[i, :, index] = 0
            if invalid_indexes.get('source_copy', None) is not None:
                source_copy_invalid_indexes = invalid_indexes['source_copy']
                for i, indexes in enumerate(source_copy_invalid_indexes):
                    for index in indexes:
                        scaled_copy_source_probs[i, :, index] = 0
        probs = torch.cat([scaled_vocab_probs.contiguous(), scaled_copy_source_probs.contiguous(), scaled_copy_target_probs.contiguous()], dim=2)
        _probs = probs.clone()
        _probs[:, :, self.vocab_size + source_dynamic_vocab_size] = 0
        _, predictions = _probs.max(2)
        return dict(probs=probs, predictions=predictions, source_dynamic_vocab_size=source_dynamic_vocab_size, target_dynamic_vocab_size=target_dynamic_vocab_size)

    def compute_loss(self, probs, predictions, generate_targets, source_copy_targets, source_dynamic_vocab_size, target_copy_targets, target_dynamic_vocab_size, coverage_records, copy_attentions):
        """
        Priority: target_copy > source_copy > generate

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param generate_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param source_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param source_dynamic_vocab_size: int
        :param target_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param target_dynamic_vocab_size: int
        :param coverage_records: None or a tensor recording source-side coverages.
            [batch_size, num_target_nodes, num_source_nodes]
        :param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
        """
        non_pad_mask = generate_targets.ne(self.vocab_pad_idx)
        source_copy_mask = source_copy_targets.ne(1) & source_copy_targets.ne(0)
        non_source_copy_mask = 1 - source_copy_mask
        target_copy_mask = target_copy_targets.ne(0)
        non_target_copy_mask = 1 - target_copy_mask
        target_copy_targets_with_offset = target_copy_targets.unsqueeze(2) + self.vocab_size + source_dynamic_vocab_size
        target_copy_target_probs = probs.gather(dim=2, index=target_copy_targets_with_offset).squeeze(2)
        target_copy_target_probs = target_copy_target_probs.mul(target_copy_mask.float())
        source_copy_targets_with_offset = source_copy_targets.unsqueeze(2) + self.vocab_size
        source_copy_target_probs = probs.gather(dim=2, index=source_copy_targets_with_offset).squeeze(2)
        source_copy_target_probs = source_copy_target_probs.mul(non_target_copy_mask.float()).mul(source_copy_mask.float())
        generate_target_probs = probs.gather(dim=2, index=generate_targets.unsqueeze(2)).squeeze(2)
        likelihood = target_copy_target_probs + source_copy_target_probs + generate_target_probs.mul(non_target_copy_mask.float()).mul(non_source_copy_mask.float())
        num_tokens = non_pad_mask.sum().item()
        if not self.force_copy:
            non_generate_oov_mask = generate_targets.ne(1)
            additional_generate_mask = non_target_copy_mask & source_copy_mask & non_generate_oov_mask
            likelihood = likelihood + generate_target_probs.mul(additional_generate_mask.float())
            num_tokens += additional_generate_mask.sum().item()
        likelihood = likelihood + self.eps
        coverage_loss = 0
        if coverage_records is not None:
            coverage_loss = torch.sum(torch.min(coverage_records, copy_attentions), 2).mul(non_pad_mask.float())
        loss = -likelihood.log().mul(non_pad_mask.float()) + coverage_loss
        targets = target_copy_targets_with_offset.squeeze(2) * target_copy_mask.long() + source_copy_targets_with_offset.squeeze(2) * non_target_copy_mask.long() * source_copy_mask.long() + generate_targets * non_target_copy_mask.long() * non_source_copy_mask.long()
        targets = targets * non_pad_mask.long()
        pred_eq = predictions.eq(targets).mul(non_pad_mask)
        num_non_pad = non_pad_mask.sum().item()
        num_correct_pred = pred_eq.sum().item()
        num_target_copy = target_copy_mask.mul(non_pad_mask).sum().item()
        num_correct_target_copy = pred_eq.mul(target_copy_mask).sum().item()
        num_correct_target_point = predictions.ge(self.vocab_size + source_dynamic_vocab_size).mul(target_copy_mask).mul(non_pad_mask).sum().item()
        num_source_copy = source_copy_mask.mul(non_target_copy_mask).mul(non_pad_mask).sum().item()
        num_correct_source_copy = pred_eq.mul(non_target_copy_mask).mul(source_copy_mask).sum().item()
        num_correct_source_point = predictions.ge(self.vocab_size).mul(predictions.lt(self.vocab_size + source_dynamic_vocab_size)).mul(non_target_copy_mask).mul(source_copy_mask).mul(non_pad_mask).sum().item()
        self.metrics(loss.sum().item(), num_non_pad, num_correct_pred, num_source_copy, num_correct_source_copy, num_correct_source_point, num_target_copy, num_correct_target_copy, num_correct_target_point)
        return dict(loss=loss.sum().div(float(num_tokens)), total_loss=loss.sum(), num_tokens=torch.tensor([float(num_tokens)]).type_as(loss), predictions=predictions)


DEFAULT_PREDICTORS = {'Seq2Seq': 'Seq2Seq', 'DeepBiaffineParser': 'BiaffineParser', 'STOG': 'STOG'}


class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """

    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) ->None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self) ->Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError('For a lazy dataset reader, _read() must return a generator')
        return instances


class DatasetReader(Registrable):
    """
    A ``DatasetReader`` knows how to turn a file containing a dataset into a collection
    of ``Instance`` s.  To implement your own, just override the `_read(file_path)` method
    to return an ``Iterable`` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to _read the data apart from the filepath should be passed
    to the constructor of the ``DatasetReader``.

    Parameters
    ----------
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """

    def __init__(self, lazy: bool=False) ->None:
        self.lazy = lazy

    def read(self, file_path: str) ->Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning('DatasetReader.lazy is not set, did you forget to call the superclass constructor?')
        if lazy:
            return _LazyInstances(lambda : iter(self._read(file_path)))
        else:
            instances = self._read(file_path)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError('No instances were read from the given filepath {}. Is the path correct?'.format(file_path))
            return instances

    def _read(self, file_path: str) ->Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) ->Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~stog.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError


JsonDict = Dict[str, Any]


class AMR:

    def __init__(self, id=None, sentence=None, graph=None, tokens=None, lemmas=None, pos_tags=None, ner_tags=None, abstract_map=None, misc=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.abstract_map = abstract_map
        self.misc = misc

    def is_named_entity(self, index):
        return self.ner_tags[index] not in ('0', 'O')

    def get_named_entity_span(self, index):
        if self.ner_tags is None or not self.is_named_entity(index):
            return []
        span = [index]
        tag = self.ner_tags[index]
        prev = index - 1
        while prev > 0 and self.ner_tags[prev] == tag:
            span.append(prev)
            prev -= 1
        next = index + 1
        while next < len(self.ner_tags) and self.ner_tags[next] == tag:
            span.append(next)
            next += 1
        return span

    def find_span_indexes(self, span):
        for i, token in enumerate(self.tokens):
            if token == span[0]:
                _span = self.tokens[i:i + len(span)]
                if len(_span) == len(span) and all(x == y for x, y in zip(span, _span)):
                    return list(range(i, i + len(span)))
        return None

    def replace_span(self, indexes, new, pos=None, ner=None):
        self.tokens = self.tokens[:indexes[0]] + new + self.tokens[indexes[-1] + 1:]
        self.lemmas = self.lemmas[:indexes[0]] + new + self.lemmas[indexes[-1] + 1:]
        if pos is None:
            pos = [self.pos_tags[indexes[0]]]
        self.pos_tags = self.pos_tags[:indexes[0]] + pos + self.pos_tags[indexes[-1] + 1:]
        if ner is None:
            ner = [self.ner_tags[indexes[0]]]
        self.ner_tags = self.ner_tags[:indexes[0]] + ner + self.ner_tags[indexes[-1] + 1:]

    def remove_span(self, indexes):
        self.replace_span(indexes, [], [], [])

    def __repr__(self):
        fields = []
        for k, v in dict(id=self.id, snt=self.sentence, tokens=self.tokens, lemmas=self.lemmas, pos_tags=self.pos_tags, ner_tags=self.ner_tags, abstract_map=self.abstract_map, misc=self.misc, graph=self.graph).items():
            if v is None:
                continue
            if k == 'misc':
                fields += v
            elif k == 'graph':
                fields.append(str(v))
            else:
                if not isinstance(v, str):
                    v = json.dumps(v)
                fields.append('# ::{} {}'.format(k, v))
        return '\n'.join(fields)

    def get_src_tokens(self):
        return self.lemmas if self.lemmas else self.sentence.split()


class AMRNode:
    attribute_priority = ['instance', 'quant', 'mode', 'value', 'name', 'li', 'mod', 'frequency', 'month', 'day', 'year', 'time', 'unit', 'decade', 'poss']

    def __init__(self, identifier, attributes=None, copy_of=None):
        self.identifier = identifier
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes
        self._num_copies = 0
        self.copy_of = copy_of

    def _sort_attributes(self):

        def get_attr_priority(attr):
            if attr in self.attribute_priority:
                return self.attribute_priority.index(attr), attr
            if not re.search('^(ARG|op|snt)', attr):
                return len(self.attribute_priority), attr
            else:
                return len(self.attribute_priority) + 1, attr
        self.attributes.sort(key=lambda x: get_attr_priority(x[0]))

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, AMRNode):
            return False
        return self.identifier == other.identifier

    def __repr__(self):
        ret = str(self.identifier)
        for k, v in self.attributes:
            if k == 'instance':
                ret += ' / ' + v
                break
        return ret

    def __str__(self):
        ret = repr(self)
        for key, value in self.attributes:
            if key == 'instance':
                continue
            ret += '\n\t:{} {}'.format(key, value)
        return ret

    @property
    def instance(self):
        for key, value in self.attributes:
            if key == 'instance':
                return value
        else:
            return None

    @property
    def ops(self):
        ops = []
        for key, value in self.attributes:
            if re.search('op\\d+', key):
                ops.append((int(key[2:]), value))
        if len(ops):
            ops.sort(key=lambda x: x[0])
        return [v for k, v in ops]

    def copy(self):
        attributes = None
        if self.attributes is not None:
            attributes = self.attributes[:]
        self._num_copies += 1
        copy = AMRNode(self.identifier + '_copy_{}'.format(self._num_copies), attributes, self)
        return copy

    def remove_attribute(self, attr, value):
        self.attributes.remove((attr, value))

    def add_attribute(self, attr, value):
        self.attributes.append((attr, value))

    def replace_attribute(self, attr, old, new):
        index = self.attributes.index((attr, old))
        self.attributes[index] = attr, new

    def get_frame_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and re.search('-\\d\\d$', v):
                yield k, v

    def get_senseless_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and not re.search('-\\d\\d$', v):
                yield k, v


def is_similar(instances1, instances2):
    if len(instances1) < len(instances2):
        small = instances1
        large = instances2
    else:
        small = instances2
        large = instances1
    coverage1 = sum(1 for x in small if x in large) / len(small)
    coverage2 = sum(1 for x in large if x in small) / len(large)
    return coverage1 > 0.8 and coverage2 > 0.8


class GraphRepair:

    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.repaired_items = set()

    @staticmethod
    def do(graph, nodes):
        gr = GraphRepair(graph, nodes)
        gr.remove_redundant_edges()
        gr.remove_unknown_nodes()

    def remove_unknown_nodes(self):
        graph = self.graph
        nodes = [node for node in graph.get_nodes()]
        for node in nodes:
            for attr, value in node.attributes:
                if value == '@@UNKNOWN@@' and attr != 'instance':
                    graph.remove_node_attribute(node, attr, value)
            if node.instance == '@@UNKNOWN@@':
                if len(list(graph._G.edges(node))) == 0:
                    for source, target in list(graph._G.in_edges(node)):
                        graph.remove_edge(source, target)
                    graph.remove_node(node)
                    self.repaired_items.add('remove-unknown-node')

    def remove_redundant_edges(self):
        """
        Edge labels such as ARGx, ARGx-of, and 'opx' should only appear at most once
        in each node's outgoing edges.
        """
        graph = self.graph
        nodes = [node for node in graph.get_nodes()]
        removed_nodes = set()
        for node in nodes:
            if node in removed_nodes:
                continue
            edges = list(graph._G.edges(node))
            edge_counter = defaultdict(list)
            for source, target in edges:
                label = graph._G[source][target]['label']
                if label == 'name':
                    edge_counter[label].append(target)
                elif label.startswith('op') or label.startswith('snt'):
                    edge_counter[str(target.instance)].append(target)
                else:
                    edge_counter[label + str(target.instance)].append(target)
            for label, children in edge_counter.items():
                if len(children) == 1:
                    continue
                if label == 'name':
                    for target in children[1:]:
                        if len(list(graph._G.in_edges(target))) == 1 and len(list(graph._G.edges(target))) == 0:
                            graph.remove_edge(node, target)
                            graph.remove_node(target)
                            removed_nodes.add(target)
                            self.repaired_items.add('remove-redundant-edge')
                    continue
                visited_children = set()
                groups = []
                for i, target in enumerate(children):
                    if target in visited_children:
                        continue
                    subtree_instances1 = [n.instance for n in graph.get_subtree(target, 5)]
                    group = [(target, subtree_instances1)]
                    visited_children.add(target)
                    for _t in children[i + 1:]:
                        if _t in visited_children or target.instance != _t.instance:
                            continue
                        subtree_instances2 = [n.instance for n in graph.get_subtree(_t, 5)]
                        if is_similar(subtree_instances1, subtree_instances2):
                            group.append((_t, subtree_instances2))
                            visited_children.add(_t)
                    groups.append(group)
                for group in groups:
                    if len(group) == 1:
                        continue
                    kept_target, _ = max(group, key=lambda x: len(x[1]))
                    for target, _ in group:
                        if target == kept_target:
                            continue
                        graph.remove_edge(node, target)
                        removed_nodes.update(graph.remove_subtree(target))


class SourceCopyVocabulary:

    def __init__(self, sentence, pad_token=DEFAULT_PADDING_TOKEN, unk_token=DEFAULT_OOV_TOKEN):
        if type(sentence) is not list:
            sentence = sentence.split(' ')
        self.src_tokens = sentence
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_token = {(0): self.pad_token, (1): self.unk_token}
        self.vocab_size = 2
        for token in sentence:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1

    def get_token_from_idx(self, idx):
        return self.idx_to_token[idx]

    def get_token_idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def index_sequence(self, list_tokens):
        return [self.get_token_idx(token) for token in list_tokens]

    def get_copy_map(self, list_tokens):
        src_indices = [self.get_token_idx(self.unk_token)] + self.index_sequence(list_tokens)
        return [(src_idx, src_token_idx) for src_idx, src_token_idx in enumerate(src_indices)]

    def get_special_tok_list(self):
        return [self.pad_token, self.unk_token]

    def __repr__(self):
        return json.dumps(self.idx_to_token)


def find_similar_token(token, tokens):
    token = re.sub('-\\d\\d$', '', token)
    for i, t in enumerate(tokens):
        if token == t:
            return tokens[i]
    return None


def is_abstract_token(token):
    return re.search('^([A-Z]+_)+\\d+$', token) or re.search('^\\d0*$', token)


def is_english_punct(c):
    return re.search('^[,.?!:;"\\\'-(){}\\[\\]]$', c)


class AMRIO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            amr = AMR()
            graph_lines = []
            misc_lines = []
            for line in f:
                line = line.rstrip()
                if line == '':
                    if len(graph_lines) != 0:
                        amr.graph = AMRGraph.decode(' '.join(graph_lines))
                        amr.graph.set_src_tokens(amr.get_src_tokens())
                        amr.misc = misc_lines
                        yield amr
                        amr = AMR()
                    graph_lines = []
                    misc_lines = []
                elif line.startswith('# ::'):
                    if line.startswith('# ::id '):
                        amr.id = line[len('# ::id '):]
                    elif line.startswith('# ::snt '):
                        amr.sentence = line[len('# ::snt '):]
                    elif line.startswith('# ::tokens '):
                        amr.tokens = json.loads(line[len('# ::tokens '):])
                    elif line.startswith('# ::lemmas '):
                        amr.lemmas = json.loads(line[len('# ::lemmas '):])
                    elif line.startswith('# ::pos_tags '):
                        amr.pos_tags = json.loads(line[len('# ::pos_tags '):])
                    elif line.startswith('# ::ner_tags '):
                        amr.ner_tags = json.loads(line[len('# ::ner_tags '):])
                    elif line.startswith('# ::abstract_map '):
                        amr.abstract_map = json.loads(line[len('# ::abstract_map '):])
                    else:
                        misc_lines.append(line)
                else:
                    graph_lines.append(line)
            if len(graph_lines) != 0:
                amr.graph = AMRGraph.decode(' '.join(graph_lines))
                amr.graph.set_src_tokens(amr.get_src_tokens())
                amr.misc = misc_lines
                yield amr

    @staticmethod
    def dump(amr_instances, f):
        for amr in amr_instances:
            f.write(str(amr) + '\n\n')


class SequenceField(Field[DataArray]):
    """
    A ``SequenceField`` represents a sequence of things.  This class just adds a method onto
    ``Field``: :func:`sequence_length`.  It exists so that ``SequenceLabelField``, ``IndexField`` and other
    similar ``Fields`` can have a single type to require, with a consistent API, whether they are
    pointing to words in a ``TextField``, items in a ``ListField``, or something else.
    """

    def sequence_length(self) ->int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError

    def empty_field(self) ->'SequenceField':
        raise NotImplementedError


START_SYMBOL = '@start@'


def pad_sequence_to_length(sequence: List, desired_length: int, default_value: Callable[[], Any]=lambda : 0, padding_on_right: bool=True) ->List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence


TokenType = TypeVar('TokenType', int, List[int])


class TokenIndexer(Generic[TokenType], Registrable):
    """
    A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.
    This class both converts strings into numerical values, with the help of a
    :class:`~stog.data.vocabulary.Vocabulary`, and it produces actual arrays.

    Tokens can be represented as single IDs (e.g., the word "cat" gets represented by the number
    34), or as lists of character IDs (e.g., "cat" gets represented by the numbers [23, 10, 18]),
    or in some other way that you can come up with (e.g., if you have some structured input you
    want to represent in a special way in your data arrays, you can do that here).
    """
    default_implementation = 'single_id'

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        """
        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
        data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
        token).  This method takes a token and a dictionary of counts and increments counts for
        whatever vocabulary items are present in the token.  If this is a single token ID
        representation, the vocabulary item is likely the token itself.  If this is a token
        characters representation, the vocabulary items are all of the characters in the token.
        """
        raise NotImplementedError

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary, index_name: str) ->Dict[str, List[TokenType]]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        This could be just an ID for each token from the vocabulary.
        Or it could split each token into characters and return one ID per character.
        Or (for instance, in the case of byte-pair encoding) there might not be a clean
        mapping from individual tokens to indices.
        """
        raise NotImplementedError

    def get_padding_token(self) ->TokenType:
        """
        When we need to add padding tokens, what should they look like?  This method returns a
        "blank" token of whatever type is returned by :func:`tokens_to_indices`.
        """
        raise NotImplementedError

    def get_padding_lengths(self, token: TokenType) ->Dict[str, int]:
        """
        This method returns a padding dictionary for the given token that specifies lengths for
        all arrays that need padding.  For example, for single ID tokens the returned dictionary
        will be empty, but for a token characters representation, this will return the number
        of characters in the token.
        """
        raise NotImplementedError

    def pad_token_sequence(self, tokens: Dict[str, List[TokenType]], desired_num_tokens: Dict[str, int], padding_lengths: Dict[str, int]) ->Dict[str, List[TokenType]]:
        """
        This method pads a list of tokens to ``desired_num_tokens`` and returns a padded copy of the
        input tokens.  If the input token list is longer than ``desired_num_tokens`` then it will be
        truncated.

        ``padding_lengths`` is used to provide supplemental padding parameters which are needed
        in some cases.  For example, it contains the widths to pad characters to when doing
        character-level padding.
        """
        raise NotImplementedError

    def get_keys(self, index_name: str) ->List[str]:
        """
        Return a list of the keys this indexer return from ``tokens_to_indices``.
        """
        return [index_name]


TokenList = List[TokenType]


def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]], remove_trailing_dimension: bool=False) ->Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.
    Parameters
    ----------
    tensor_dicts : ``List[Dict[str, torch.Tensor]]``
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : ``bool``
        If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
        batched, and remove it if we find it.
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        if remove_trailing_dimension and all(tensor.size(-1) == 1 for tensor in tensor_list):
            batched_tensor = batched_tensor.squeeze(-1)
        batched_tensors[key] = batched_tensor
    return batched_tensors


def load_dataset_reader(dataset_type, *args, **kwargs):
    if dataset_type == 'AMR':
        dataset_reader = AbstractMeaningRepresentationDatasetReader(token_indexers=dict(encoder_tokens=SingleIdTokenIndexer(namespace='encoder_token_ids'), encoder_characters=TokenCharactersIndexer(namespace='encoder_token_characters'), decoder_tokens=SingleIdTokenIndexer(namespace='decoder_token_ids'), decoder_characters=TokenCharactersIndexer(namespace='decoder_token_characters')), word_splitter=kwargs.get('word_splitter', None))
    else:
        raise NotImplementedError
    return dataset_reader


def sanitize(x: Any) ->Any:
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        return x.tolist()
    elif isinstance(x, numpy.number):
        return x.item()
    elif isinstance(x, dict):
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple)):
        return [sanitize(x_i) for x_i in x]
    elif isinstance(x, (spacy.tokens.Token, allennlp.data.Token)):
        return x.text
    elif x is None:
        return 'None'
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f'Cannot sanitize {x} of type {type(x)}. If this is your own custom class, add a `to_json(self)` method that returns a JSON-like object.')


class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model, dataset_reader: DatasetReader) ->None:
        self._model = model
        self._dataset_reader = dataset_reader

    def load_line(self, line: str) ->JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) ->str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + '\n'

    def predict_json(self, inputs: JsonDict) ->JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict_instance(self, instance: Instance) ->JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _json_to_instance(self, json_dict: JsonDict) ->Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[JsonDict]) ->List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) ->List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) ->List[Instance]:
        """
        Converts a list of JSON objects into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by :func:`predict_json`. In order to use this method for
        batch prediction, :func:`_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(cls, archive_path: str, predictor_name: str=None) ->'Predictor':
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as running the predictor on the GPU,
        please use `from_archive`.

        Parameters
        ----------
        archive_path The path to the archive.

        Returns
        -------
        A Predictor instance.
        """
        raise NotImplementedError

    @classmethod
    def from_archive(cls, archive, predictor_name: str=None) ->'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        config = archive.config.duplicate()
        if not predictor_name:
            model_type = config.get('model').get('model_type')
            if not model_type in DEFAULT_PREDICTORS:
                raise ConfigurationError(f'No default predictor for model type {model_type}.\nPlease specify a predictor explicitly.')
            predictor_name = DEFAULT_PREDICTORS[model_type]
        word_splitter = None
        if config['model'].get('use_bert', False):
            word_splitter = config['data'].get('word_splitter', None)
        dataset_reader = load_dataset_reader(config['data']['data_type'], word_splitter=word_splitter)
        if hasattr(dataset_reader, 'set_evaluation'):
            dataset_reader.set_evaluation()
        model = archive.model
        model.eval()
        return Predictor.by_name(predictor_name)(model, dataset_reader)


class Seq2SeqEncoder(_EncoderBase):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/modules/seq2seq_encoders/seq2seq_encoder.py

    A ``Seq2SeqEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape: ``(batch_size, sequence_length, input_dim)``; output
    shape: ``(batch_size, sequence_length, output_dim)``.
    We add two methods to the basic ``Module`` API: :func:`get_input_dim()` and :func:`get_output_dim()`.
    You might need this if you want to construct a ``Linear`` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """

    def get_input_dim(self) ->int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) ->int:
        """
        Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_bidirectional(self) ->bool:
        """
        Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
        of the encoder is the first half of the final dimension, and the backward direction is the
        second half.
        """
        raise NotImplementedError


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int], gain: float=1.0) ->None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([(a % b != 0) for a, b in zip(sizes, split_sizes)]):
        raise ConfigurationError('tensor dimensions must be divisible by their respective split_sizes. Found size: {} and split_sizes: {}'.format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step) for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.new_tensor(torch.rand(tensor_for_masking.size()) > dropout_probability)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class AugmentedLstm(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers. Note: this implementation is slower
    than the native Pytorch LSTM because it cannot make use of CUDNN
    optimizations for stacked RNNs due to the highway layers and
    variational dropout.
    Parameters
    ----------
    input_size : int, required.
        The dimension of the inputs to the LSTM.
    hidden_size : int, required.
        The dimension of the outputs of the LSTM.
    go_forward: bool, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::
            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.
    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """

    def __init__(self, input_size: int, hidden_size: int, go_forward: bool=True, recurrent_dropout_probability: float=0.0, use_highway: bool=True, use_input_projection_bias: bool=True) ->None:
        super(AugmentedLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability
        if use_highway:
            self.input_linearity = torch.nn.Linear(input_size, 6 * hidden_size, bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(input_size, 4 * hidden_size, bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, inputs: PackedSequence, initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Parameters
        ----------
        inputs : PackedSequence, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).
        Returns
        -------
        A PackedSequence containing a torch.FloatTensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the LSTM per timestep and a tuple containing
        the LSTM state, with shape (1, batch_size, hidden_size) to
        match the Pytorch API.
        """
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % type(inputs))
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.hidden_size)
        if initial_state is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
            full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_memory)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:current_length_index + 1].clone()
            timestep_input = sequence_tensor[0:current_length_index + 1, index]
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] + projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] + projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] + projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] + projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * torch.tanh(memory)
            if self.use_highway:
                highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] + projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
                highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
                timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection
            if dropout_mask is not None and self.training:
                timestep_output = timestep_output * dropout_mask[0:current_length_index + 1]
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output
        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        final_state = full_batch_previous_state.unsqueeze(0), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


class StackedBidirectionalLstm(torch.nn.Module):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/modules/stacked_bidirectional_lstm.py

    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float=0.0, use_highway: bool=True) ->None:
        super(StackedBidirectionalLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            forward_layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward=True, recurrent_dropout_probability=recurrent_dropout_probability, use_highway=use_highway, use_input_projection_bias=False)
            backward_layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward=False, recurrent_dropout_probability=recurrent_dropout_probability, use_highway=use_highway, use_input_projection_bias=False)
            lstm_input_size = hidden_size * 2
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            layers.append([forward_layer, backward_layer])
        self.lstm_layers = layers

    def forward(self, inputs: PackedSequence, initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension * 2).
        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size * 2).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError('Initial states were passed to forward() but the number of initial states does not match the number of layers.')
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(i))
            backward_layer = getattr(self, 'backward_layer_{}'.format(i))
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            backward_output, final_backward_state = backward_layer(output_sequence, state)
            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)
            final_states.append(torch.cat(both_direction_states, -1) for both_direction_states in zip(final_forward_state, final_backward_state))
        final_state_tuple = [torch.cat(state_list, 0) for state_list in zip(*final_states)]
        return output_sequence, final_state_tuple

    @classmethod
    def from_params(cls, params):
        return cls(input_size=params['input_size'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], recurrent_dropout_probability=params.get('dropout', 0.0), use_highway=params.get('use_highway', True))


class StackedLstm(torch.nn.Module):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/modules/stacked_bidirectional_lstm.py

    A standard stacked LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float=0.0, use_highway: bool=True) ->None:
        super(StackedLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward=True, recurrent_dropout_probability=recurrent_dropout_probability, use_highway=use_highway, use_input_projection_bias=False)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self, inputs: PackedSequence, initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).
        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError('Initial states were passed to forward() but the number of initial states does not match the number of layers.')
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'layer_{}'.format(i))
            output_sequence, final_state = forward_layer(output_sequence, state)
            final_states.append(final_state)
        final_state_tuple = [torch.cat(state_list, 0) for state_list in zip(*final_states)]
        return output_sequence, final_state_tuple

    @classmethod
    def from_params(cls, params):
        return cls(input_size=params['input_size'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], recurrent_dropout_probability=params.get('dropout', 0.0), use_highway=params.get('use_highway', True))


def lazy_groups_of(iterator: Iterator[A], group_size: int) ->Iterator[List[A]]:
    """
    Takes an iterator and batches the invididual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    return iter(lambda : list(islice(iterator, 0, group_size)), [])


class _PredictManager:

    def __init__(self, predictor: Predictor, input_file: str, output_file: Optional[str], batch_size: int, print_to_console: bool, has_dataset_reader: bool, beam_size: int) ->None:
        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, 'w')
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader
        else:
            self._dataset_reader = None
        if type(predictor) in (STOGPredictor,):
            self.beam_size = beam_size
            self._predictor._model.set_beam_size(self.beam_size)
            self._predictor._model.set_decoder_token_indexers(self._dataset_reader._token_indexers)

    def _predict_json(self, batch_data: List[JsonDict]) ->Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance]) ->Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(self, prediction: str, model_input: str=None) ->None:
        if self._print_to_console:
            if model_input is not None:
                None
            None
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self) ->Iterator[JsonDict]:
        if self._input_file == '-':
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            with open(self._input_file, 'r') as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) ->Iterator[Instance]:
        if self._input_file == '-':
            raise ConfigurationError('stdin is not an option when using a DatasetReader.')
        elif self._dataset_reader is None:
            raise ConfigurationError('To generate instances directly, pass a DatasetReader.')
        else:
            yield from self._dataset_reader.read(self._input_file)

    def run(self) ->None:
        has_reader = self._dataset_reader is not None
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(result, str(model_input_instance))
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(result, json.dumps(model_input_json))
        if self._output_file is not None:
            self._output_file.close()


def character_tensor_from_token_tensor(token_tensor, vocab, character_tokenizer, namespace=dict(tokens='decoder_token_ids', characters='decoder_token_characters')):
    token_str = [vocab.get_token_from_index(i, namespace['tokens']) for i in token_tensor.view(-1).tolist()]
    max_char_len = max([len(token) for token in token_str])
    indices = []
    for token in token_str:
        token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
        for char_i, character in enumerate(character_tokenizer.tokenize(token)):
            index = vocab.get_token_index(character.text, namespace['characters'])
            token_indices[char_i] = index
        indices.append(token_indices)
    return torch.tensor(indices).view(token_tensor.size(0), token_tensor.size(1), -1).type_as(token_tensor)


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor], num_wrapping_dims: int=0) ->torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.
    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.
    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.
    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.
    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.V(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if 'mask' in text_field_tensors:
        return text_field_tensors['mask']
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])
    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError('Expected a tensor with dimension 2 or 3, found {}'.format(smallest_dim))


class STOG(Model):

    def __init__(self, vocab, punctuation_ids, use_must_copy_embedding, use_char_cnn, use_coverage, use_aux_encoder, use_bert, max_decode_length, bert_encoder, encoder_token_embedding, encoder_pos_embedding, encoder_must_copy_embedding, encoder_char_embedding, encoder_char_cnn, encoder_embedding_dropout, encoder, encoder_output_dropout, decoder_token_embedding, decoder_pos_embedding, decoder_coref_embedding, decoder_char_embedding, decoder_char_cnn, decoder_embedding_dropout, decoder, aux_encoder, aux_encoder_output_dropout, generator, graph_decoder, test_config):
        super(STOG, self).__init__()
        self.vocab = vocab
        self.punctuation_ids = punctuation_ids
        self.use_must_copy_embedding = use_must_copy_embedding
        self.use_char_cnn = use_char_cnn
        self.use_coverage = use_coverage
        self.use_aux_encoder = use_aux_encoder
        self.use_bert = use_bert
        self.max_decode_length = max_decode_length
        self.bert_encoder = bert_encoder
        self.encoder_token_embedding = encoder_token_embedding
        self.encoder_pos_embedding = encoder_pos_embedding
        self.encoder_must_copy_embedding = encoder_must_copy_embedding
        self.encoder_char_embedding = encoder_char_embedding
        self.encoder_char_cnn = encoder_char_cnn
        self.encoder_embedding_dropout = encoder_embedding_dropout
        self.encoder = encoder
        self.encoder_output_dropout = encoder_output_dropout
        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_pos_embedding = decoder_pos_embedding
        self.decoder_coref_embedding = decoder_coref_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_char_cnn = decoder_char_cnn
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.decoder = decoder
        self.aux_encoder = aux_encoder
        self.aux_encoder_output_dropout = aux_encoder_output_dropout
        self.generator = generator
        self.graph_decoder = graph_decoder
        self.beam_size = 1
        self.test_config = test_config

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_decoder_token_indexers(self, token_indexers):
        self.decoder_token_indexers = token_indexers
        self.character_tokenizer = CharacterTokenizer()

    def get_metrics(self, reset: bool=False, mimick_test: bool=False):
        metrics = dict()
        if mimick_test and self.test_config:
            metrics = self.mimick_test()
        generator_metrics = self.generator.metrics.get_metric(reset)
        graph_decoder_metrics = self.graph_decoder.metrics.get_metric(reset)
        metrics.update(generator_metrics)
        metrics.update(graph_decoder_metrics)
        if 'F1' not in metrics:
            metrics['F1'] = metrics['all_acc']
        return metrics

    def mimick_test(self):
        word_splitter = None
        if self.use_bert:
            word_splitter = self.test_config.get('word_splitter', None)
        dataset_reader = load_dataset_reader('AMR', word_splitter=word_splitter)
        dataset_reader.set_evaluation()
        predictor = Predictor.by_name('STOG')(self, dataset_reader)
        manager = _PredictManager(predictor, self.test_config['data'], self.test_config['prediction'], self.test_config['batch_size'], False, True, 1)
        try:
            logger.info('Mimicking test...')
            manager.run()
        except Exception as e:
            logger.info('Exception threw out when running the manager.')
            logger.error(e, exc_info=True)
            return {}
        try:
            logger.info('Computing the Smatch score...')
            result = subprocess.check_output([self.test_config['eval_script'], self.test_config['smatch_dir'], self.test_config['data'], self.test_config['prediction']]).decode().split()
            result = list(map(float, result))
            return dict(PREC=result[0] * 100, REC=result[1] * 100, F1=result[2] * 100)
        except Exception as e:
            logger.info('Exception threw out when computing smatch.')
            logger.error(e, exc_info=True)
            return {}

    def print_batch_details(self, batch, batch_idx):
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None

    def prepare_batch_input(self, batch):
        bert_token_inputs = batch.get('src_token_ids', None)
        if bert_token_inputs is not None:
            bert_token_inputs = bert_token_inputs.long()
        encoder_token_subword_index = batch.get('src_token_subword_index', None)
        if encoder_token_subword_index is not None:
            encoder_token_subword_index = encoder_token_subword_index.long()
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        encoder_pos_tags = batch['src_pos_tags']
        encoder_must_copy_tags = batch['src_must_copy_tags']
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        encoder_mask = get_text_field_mask(batch['src_tokens'])
        encoder_inputs = dict(bert_token=bert_token_inputs, token_subword_index=encoder_token_subword_index, token=encoder_token_inputs, pos_tag=encoder_pos_tags, must_copy_tag=encoder_must_copy_tags, char=encoder_char_inputs, mask=encoder_mask)
        decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'][:, :-1].contiguous()
        decoder_pos_tags = batch['tgt_pos_tags'][:, :-1]
        decoder_char_inputs = batch['tgt_tokens']['decoder_characters'][:, :-1].contiguous()
        raw_coref_inputs = batch['tgt_copy_indices'][:, :-1].contiguous()
        coref_happen_mask = raw_coref_inputs.ne(0)
        decoder_coref_inputs = torch.ones_like(raw_coref_inputs) * torch.arange(0, raw_coref_inputs.size(1)).type_as(raw_coref_inputs).unsqueeze(0)
        decoder_coref_inputs.masked_fill_(coref_happen_mask, 0)
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs
        decoder_inputs = dict(token=decoder_token_inputs, pos_tag=decoder_pos_tags, char=decoder_char_inputs, coref=decoder_coref_inputs)
        vocab_targets = batch['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()
        coref_targets = batch['tgt_copy_indices'][:, 1:]
        coref_attention_maps = batch['tgt_copy_map'][:, 1:]
        copy_targets = batch['src_copy_indices'][:, 1:]
        copy_attention_maps = batch['src_copy_map'][:, 1:-1]
        generator_inputs = dict(vocab_targets=vocab_targets, coref_targets=coref_targets, coref_attention_maps=coref_attention_maps, copy_targets=copy_targets, copy_attention_maps=copy_attention_maps)
        edge_heads = batch['head_indices'][:, :-2]
        edge_labels = batch['head_tags'][:, :-2]
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[parser_token_inputs == self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')] = 0
        parser_mask = (parser_token_inputs != 0).float()
        parser_inputs = dict(edge_heads=edge_heads, edge_labels=edge_labels, corefs=decoder_coref_inputs, mask=parser_mask)
        return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs

    def forward(self, batch, for_training=False):
        encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = self.prepare_batch_input(batch)
        encoder_outputs = self.encode(encoder_inputs['bert_token'], encoder_inputs['token_subword_index'], encoder_inputs['token'], encoder_inputs['pos_tag'], encoder_inputs['must_copy_tag'], encoder_inputs['char'], encoder_inputs['mask'])
        if for_training:
            decoder_outputs = self.decode_for_training(decoder_inputs['token'], decoder_inputs['pos_tag'], decoder_inputs['char'], decoder_inputs['coref'], encoder_outputs['memory_bank'], encoder_inputs['mask'], encoder_outputs['final_states'], parser_inputs['mask'])
            generator_output = self.generator(decoder_outputs['memory_bank'], decoder_outputs['copy_attentions'], generator_inputs['copy_attention_maps'], decoder_outputs['coref_attentions'], generator_inputs['coref_attention_maps'])
            generator_loss_output = self.generator.compute_loss(generator_output['probs'], generator_output['predictions'], generator_inputs['vocab_targets'], generator_inputs['copy_targets'], generator_output['source_dynamic_vocab_size'], generator_inputs['coref_targets'], generator_output['target_dynamic_vocab_size'], decoder_outputs['coverage_records'], decoder_outputs['copy_attentions'])
            graph_decoder_outputs = self.graph_decode(decoder_outputs['rnn_memory_bank'], parser_inputs['edge_heads'], parser_inputs['edge_labels'], parser_inputs['corefs'], parser_inputs['mask'], decoder_outputs['aux_encoder_outputs'])
            return dict(loss=generator_loss_output['loss'] + graph_decoder_outputs['loss'], token_loss=generator_loss_output['total_loss'], edge_loss=graph_decoder_outputs['total_loss'], num_tokens=generator_loss_output['num_tokens'], num_nodes=graph_decoder_outputs['num_nodes'])
        else:
            invalid_indexes = dict(source_copy=batch.get('source_copy_invalid_ids', None), vocab=[set(self.punctuation_ids) for _ in range(len(batch['tag_lut']))])
            return dict(encoder_memory_bank=encoder_outputs['memory_bank'], encoder_mask=encoder_inputs['mask'], encoder_final_states=encoder_outputs['final_states'], copy_attention_maps=generator_inputs['copy_attention_maps'], copy_vocabs=batch['src_copy_vocab'], tag_luts=batch['tag_lut'], invalid_indexes=invalid_indexes)

    def encode(self, bert_tokens, token_subword_index, tokens, pos_tags, must_copy_tags, chars, mask):
        encoder_inputs = []
        if self.use_bert:
            bert_mask = bert_tokens.ne(0)
            bert_embeddings, _ = self.bert_encoder(bert_tokens, attention_mask=bert_mask, output_all_encoded_layers=False, token_subword_index=token_subword_index)
            if token_subword_index is None:
                bert_embeddings = bert_embeddings[:, 1:-1]
            encoder_inputs += [bert_embeddings]
        token_embeddings = self.encoder_token_embedding(tokens)
        pos_tag_embeddings = self.encoder_pos_embedding(pos_tags)
        encoder_inputs += [token_embeddings, pos_tag_embeddings]
        if self.use_must_copy_embedding:
            must_copy_tag_embeddings = self.encoder_must_copy_embedding(must_copy_tags)
            encoder_inputs += [must_copy_tag_embeddings]
        if self.use_char_cnn:
            char_cnn_output = self._get_encoder_char_cnn_output(chars)
            encoder_inputs += [char_cnn_output]
        encoder_inputs = torch.cat(encoder_inputs, 2)
        encoder_inputs = self.encoder_embedding_dropout(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs, mask)
        encoder_outputs = self.encoder_output_dropout(encoder_outputs)
        encoder_final_states = self.encoder._states
        self.encoder.reset_states()
        return dict(memory_bank=encoder_outputs, final_states=encoder_final_states)

    def decode_for_training(self, tokens, pos_tags, chars, corefs, memory_bank, mask, states, tgt_mask):
        token_embeddings = self.decoder_token_embedding(tokens)
        pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
        coref_embeddings = self.decoder_coref_embedding(corefs)
        if self.use_char_cnn:
            char_cnn_output = self._get_decoder_char_cnn_output(chars)
            decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings, char_cnn_output], 2)
        else:
            decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings], 2)
        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, memory_bank, mask, states)
        if self.use_aux_encoder:
            aux_encoder_inputs = decoder_inputs[:, 1:]
            aux_encoder_outputs = self.aux_encoder(aux_encoder_inputs, tgt_mask[:, 1:].byte())
            aux_encoder_outputs = self.aux_encoder_output_dropout(aux_encoder_outputs)
            self.aux_encoder.reset_states()
        else:
            aux_encoder_outputs = None
        return dict(memory_bank=decoder_outputs['decoder_hidden_states'], rnn_memory_bank=decoder_outputs['rnn_hidden_states'], coref_attentions=decoder_outputs['target_copy_attentions'], copy_attentions=decoder_outputs['source_copy_attentions'], coverage_records=decoder_outputs['coverage_records'], aux_encoder_outputs=aux_encoder_outputs)

    def graph_decode(self, memory_bank, edge_heads, edge_labels, corefs, mask, aux_memory_bank):
        memory_bank = memory_bank[:, 1:]
        if self.use_aux_encoder:
            memory_bank = torch.cat([memory_bank, aux_memory_bank], 2)
        corefs = corefs[:, 1:]
        mask = mask[:, 1:]
        return self.graph_decoder(memory_bank, edge_heads, edge_labels, corefs, mask)

    def _get_encoder_char_cnn_output(self, chars):
        char_embeddings = self.encoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def _get_decoder_char_cnn_output(self, chars):
        char_embeddings = self.decoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def decode(self, input_dict):
        memory_bank = input_dict['encoder_memory_bank']
        mask = input_dict['encoder_mask']
        states = input_dict['encoder_final_states']
        copy_attention_maps = input_dict['copy_attention_maps']
        copy_vocabs = input_dict['copy_vocabs']
        tag_luts = input_dict['tag_luts']
        invalid_indexes = input_dict['invalid_indexes']
        if self.beam_size == 0:
            generator_outputs = self.decode_with_pointer_generator(memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)
        else:
            generator_outputs = self.beam_search_with_pointer_generator(memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)
        parser_outputs = self.decode_with_graph_parser(generator_outputs['decoder_inputs'], generator_outputs['decoder_rnn_memory_bank'], generator_outputs['coref_indexes'], generator_outputs['decoder_mask'])
        return dict(nodes=generator_outputs['predictions'], heads=parser_outputs['edge_heads'], head_labels=parser_outputs['edge_labels'], corefs=generator_outputs['coref_indexes'])

    def beam_search_with_pointer_generator(self, memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indices):
        batch_size = memory_bank.size(0)
        beam_size = self.beam_size
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, beam_size).view(-1).type_as(mask)
        bos_token = self.vocab.get_token_index(START_SYMBOL, 'decoder_token_ids')
        eos_token = self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        pad_token = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'decoder_token_ids')
        bucket = [[] for i in range(batch_size)]
        bucket_max_score = [(-100000000.0) for i in range(batch_size)]

        def flatten(tensor):
            sizes = list(tensor.size())
            assert len(sizes) >= 2
            assert sizes[0] == batch_size and sizes[1] == beam_size
            if len(sizes) == 2:
                new_sizes = [sizes[0] * sizes[1], 1]
            else:
                new_sizes = [sizes[0] * sizes[1]] + sizes[2:]
            return tensor.contiguous().view(new_sizes)

        def fold(tensor):
            sizes = list(tensor.size())
            new_sizes = [batch_size, beam_size]
            if len(sizes) >= 2:
                new_sizes = [batch_size, beam_size] + sizes[1:]
            return tensor.view(new_sizes)

        def beam_select_2d(input, indices):
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) >= 2
            assert input_size[0] == indices_size[0]
            assert input_size[1] == indices_size[1]
            return input.view([input_size[0] * input_size[1]] + input_size[2:]).index_select(0, (torch.arange(indices_size[0]).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices).view(-1)).view(input_size)

        def beam_select_1d(input, indices):
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) > 1
            assert input_size[0] == indices_size[0] * indices_size[1]
            return input.index_select(0, (torch.arange(indices_size[0]).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices).view(-1)).view(input_size)

        def update_tensor_buff(key, step, beam_indices, tensor, select_input=True):
            if step == 0 and beam_buffer[key] is None:
                beam_buffer[key] = tensor.new_zeros(batch_size, beam_size, self.max_decode_length, tensor.size(-1))
            if select_input:
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
            else:
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))

        def get_decoder_input(tokens, pos_tags, corefs):
            token_embeddings = self.decoder_token_embedding(tokens)
            pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            coref_embeddings = self.decoder_coref_embedding(corefs)
            if self.use_char_cnn:
                chars = character_tensor_from_token_tensor(tokens, self.vocab, self.character_tokenizer)
                if chars.size(-1) < 3:
                    chars = torch.cat((chars, chars.new_zeros((chars.size(0), chars.size(1), 3 - chars.size(2)))), 2)
                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings], 2)
            return self.decoder_embedding_dropout(decoder_inputs)

        def repeat_list_item(input_list, n):
            new_list = []
            for item in input_list:
                new_list += [item] * n
            return new_list
        beam_buffer = {}
        beam_buffer['predictions'] = mask.new_full((batch_size, beam_size, self.max_decode_length), pad_token)
        beam_buffer['coref_indexes'] = memory_bank.new_zeros(batch_size, beam_size, self.max_decode_length)
        beam_buffer['decoder_mask'] = memory_bank.new_ones(batch_size, beam_size, self.max_decode_length)
        beam_buffer['decoder_inputs'] = None
        beam_buffer['decoder_memory_bank'] = None
        beam_buffer['decoder_rnn_memory_bank'] = None
        beam_buffer['scores'] = memory_bank.new_zeros(batch_size, beam_size, 1)
        beam_buffer['scores'][:, 1:] = -float(100000000.0)
        variables = {}
        variables['input_tokens'] = beam_buffer['predictions'].new_full((batch_size * beam_size, 1), bos_token)
        variables['pos_tags'] = mask.new_full((batch_size * beam_size, 1), self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
        variables['corefs'] = mask.new_zeros(batch_size * beam_size, 1)
        variables['input_feed'] = None
        variables['coref_inputs'] = []
        variables['states'] = [item.index_select(1, new_order) for item in states]
        variables['prev_tokens'] = mask.new_full((batch_size * beam_size, 1), bos_token)
        variables['coref_attention_maps'] = memory_bank.new_zeros(batch_size * beam_size, self.max_decode_length, self.max_decode_length + 1)
        variables['coref_vocab_maps'] = mask.new_zeros(batch_size * beam_size, self.max_decode_length + 1)
        variables['coverage'] = None
        if self.use_coverage:
            variables['coverage'] = memory_bank.new_zeros(batch_size * beam_size, 1, memory_bank.size(1))
        for key in invalid_indices.keys():
            invalid_indices[key] = repeat_list_item(invalid_indices[key], beam_size)
        for step in range(self.max_decode_length):
            decoder_inputs = get_decoder_input(variables['input_tokens'], variables['pos_tags'], variables['corefs'])
            decoder_output_dict = self.decoder(decoder_inputs, memory_bank.index_select(0, new_order), mask.index_select(0, new_order), variables['states'], variables['input_feed'], variables['coref_inputs'], variables['coverage'])
            _decoder_outputs = decoder_output_dict['decoder_hidden_states']
            _rnn_outputs = decoder_output_dict['rnn_hidden_states']
            _copy_attentions = decoder_output_dict['source_copy_attentions']
            _coref_attentions = decoder_output_dict['target_copy_attentions']
            states = decoder_output_dict['last_hidden_state']
            input_feed = decoder_output_dict['input_feed']
            coverage = decoder_output_dict['coverage']
            coverage_records = decoder_output_dict['coverage_records']
            if step == 0:
                _coref_attention_maps = variables['coref_attention_maps'][:, :step + 1]
            else:
                _coref_attention_maps = variables['coref_attention_maps'][:, :step]
            generator_output = self.generator(_decoder_outputs, _copy_attentions, copy_attention_maps.index_select(0, new_order), _coref_attentions, _coref_attention_maps, invalid_indices)
            word_lprobs = fold(torch.log(1e-08 + generator_output['probs'].squeeze(1)))
            if self.use_coverage:
                coverage_loss = torch.sum(torch.min(coverage, _copy_attentions), dim=2)
            else:
                coverage_loss = word_lprobs.new_zeros(batch_size, beam_size, 1)
            new_all_scores = word_lprobs + beam_buffer['scores'].expand_as(word_lprobs) - coverage_loss.view(batch_size, beam_size, 1).expand_as(word_lprobs)
            new_hypo_scores, new_hypo_indices = torch.topk(new_all_scores.view(batch_size, -1).contiguous(), beam_size * 2, dim=-1)
            new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))
            eos_token_mask = new_token_indices.eq(eos_token)
            eos_beam_indices_offset = torch.div(new_hypo_indices, word_lprobs.size(-1))[:, :beam_size] + new_order.view(batch_size, beam_size) * beam_size
            eos_beam_indices_offset = eos_beam_indices_offset.masked_select(eos_token_mask[:, :beam_size])
            if eos_beam_indices_offset.numel() > 0:
                for index in eos_beam_indices_offset.tolist():
                    eos_batch_idx = int(index / beam_size)
                    eos_beam_idx = index % beam_size
                    hypo_score = float(new_hypo_scores[eos_batch_idx, eos_beam_idx]) / (step + 1)
                    if step > 0 and hypo_score > bucket_max_score[eos_batch_idx] and eos_beam_idx == 0:
                        bucket_max_score[eos_batch_idx] = hypo_score
                        bucket[eos_batch_idx] += [{key: tensor[eos_batch_idx, eos_beam_idx].unsqueeze(0) for key, tensor in beam_buffer.items()}]
                eos_token_mask = eos_token_mask.type_as(new_hypo_scores)
                active_hypo_scores, active_sort_indices = torch.sort((1 - eos_token_mask) * new_hypo_scores + eos_token_mask * -float(100000000.0), descending=True)
                active_sort_indices_offset = active_sort_indices + 2 * beam_size * torch.arange(batch_size).unsqueeze(1).expand_as(active_sort_indices).type_as(active_sort_indices)
                active_hypo_indices = new_hypo_indices.view(batch_size * beam_size * 2)[active_sort_indices_offset.view(batch_size * beam_size * 2)].view(batch_size, -1)
                new_hypo_scores = active_hypo_scores
                new_hypo_indices = active_hypo_indices
                new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))
            new_hypo_indices = new_hypo_indices[:, :beam_size]
            new_hypo_scores = new_hypo_scores[:, :beam_size]
            new_token_indices = new_token_indices[:, :beam_size]
            beam_indices = torch.div(new_hypo_indices, word_lprobs.size(-1))
            if step == 0:
                decoder_mask_input = []
            else:
                decoder_mask_input = beam_select_2d(beam_buffer['decoder_mask'], beam_indices).view(batch_size * beam_size, -1)[:, :step].split(1, 1)
            variables['coref_attention_maps'] = beam_select_1d(variables['coref_attention_maps'], beam_indices)
            variables['coref_vocab_maps'] = beam_select_1d(variables['coref_vocab_maps'], beam_indices)
            input_tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(step, flatten(new_token_indices).squeeze(1), generator_output['source_dynamic_vocab_size'], variables['coref_attention_maps'], variables['coref_vocab_maps'], repeat_list_item(copy_vocabs, beam_size), decoder_mask_input, repeat_list_item(tag_luts, beam_size), invalid_indices)
            beam_buffer['scores'] = new_hypo_scores.unsqueeze(2)
            update_tensor_buff('decoder_inputs', step, beam_indices, decoder_inputs)
            update_tensor_buff('decoder_memory_bank', step, beam_indices, _decoder_outputs)
            update_tensor_buff('decoder_rnn_memory_bank', step, beam_indices, _rnn_outputs)
            update_tensor_buff('predictions', step, beam_indices, _predictions, False)
            update_tensor_buff('coref_indexes', step, beam_indices, corefs, False)
            update_tensor_buff('decoder_mask', step, beam_indices, _mask, False)
            variables['input_tokens'] = input_tokens
            variables['pos_tags'] = pos_tags
            variables['corefs'] = corefs
            variables['states'] = [state.index_select(1, new_order * beam_size + beam_indices.view(-1)) for state in states]
            variables['input_feed'] = beam_select_1d(input_feed, beam_indices)
            variables['coref_inputs'] = list(beam_select_1d(torch.cat(variables['coref_inputs'], 1), beam_indices).split(1, 1))
            if self.use_coverage:
                variables['coverage'] = beam_select_1d(coverage, beam_indices)
            else:
                variables['coverage'] = None
        for batch_idx, item in enumerate(bucket):
            if len(item) == 0:
                bucket[batch_idx].append({key: tensor[batch_idx, 0].unsqueeze(0) for key, tensor in beam_buffer.items()})
        return_dict = {}
        for key in bucket[0][-1].keys():
            return_dict[key] = torch.cat([hypos[-1][key] for hypos in bucket], dim=0)
        return_dict['decoder_mask'] = 1 - return_dict['decoder_mask']
        return_dict['decoder_inputs'] = return_dict['decoder_inputs'][:, 1:]
        return_dict['decoder_memory_bank'] = return_dict['decoder_memory_bank'][:, 1:]
        return_dict['decoder_rnn_memory_bank'] = return_dict['decoder_rnn_memory_bank'][:, 1:]
        return_dict['predictions'] = return_dict['predictions'][:, :-1]
        return_dict['predictions'][return_dict['predictions'] == pad_token] = eos_token
        return_dict['coref_indexes'] = return_dict['coref_indexes'][:, :-1]
        return_dict['decoder_mask'] = return_dict['predictions'] != eos_token
        return_dict['scores'] = torch.div(return_dict['scores'], return_dict['decoder_mask'].sum(1, keepdim=True).type_as(return_dict['scores']))
        return return_dict

    def decode_with_pointer_generator(self, memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes):
        batch_size = memory_bank.size(0)
        tokens = torch.ones(batch_size, 1) * self.vocab.get_token_index(START_SYMBOL, 'decoder_token_ids')
        pos_tags = torch.ones(batch_size, 1) * self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags')
        tokens = tokens.type_as(mask).long()
        pos_tags = pos_tags.type_as(tokens)
        corefs = torch.zeros(batch_size, 1).type_as(mask).long()
        decoder_input_history = []
        decoder_outputs = []
        rnn_outputs = []
        copy_attentions = []
        coref_attentions = []
        predictions = []
        coref_indexes = []
        decoder_mask = []
        input_feed = None
        coref_inputs = []
        coref_attention_maps = torch.zeros(batch_size, self.max_decode_length, self.max_decode_length + 1).type_as(memory_bank)
        coref_vocab_maps = torch.zeros(batch_size, self.max_decode_length + 1).type_as(mask).long()
        coverage = None
        if self.use_coverage:
            coverage = memory_bank.new_zeros(batch_size, 1, memory_bank.size(1))
        for step_i in range(self.max_decode_length):
            token_embeddings = self.decoder_token_embedding(tokens)
            pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            coref_embeddings = self.decoder_coref_embedding(corefs)
            if self.use_char_cnn:
                chars = character_tensor_from_token_tensor(tokens, self.vocab, self.character_tokenizer)
                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = torch.cat([token_embeddings, pos_tag_embeddings, coref_embeddings], 2)
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)
            decoder_output_dict = self.decoder(decoder_inputs, memory_bank, mask, states, input_feed, coref_inputs, coverage)
            _decoder_outputs = decoder_output_dict['decoder_hidden_states']
            _rnn_outputs = decoder_output_dict['rnn_hidden_states']
            _copy_attentions = decoder_output_dict['source_copy_attentions']
            _coref_attentions = decoder_output_dict['target_copy_attentions']
            states = decoder_output_dict['last_hidden_state']
            input_feed = decoder_output_dict['input_feed']
            coverage = decoder_output_dict['coverage']
            if step_i == 0:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1]
            else:
                _coref_attention_maps = coref_attention_maps[:, :step_i]
            generator_output = self.generator(_decoder_outputs, _copy_attentions, copy_attention_maps, _coref_attentions, _coref_attention_maps, invalid_indexes)
            _predictions = generator_output['predictions']
            tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(step_i, generator_output['predictions'].squeeze(1), generator_output['source_dynamic_vocab_size'], coref_attention_maps, coref_vocab_maps, copy_vocabs, decoder_mask, tag_luts, invalid_indexes)
            decoder_input_history += [decoder_inputs]
            decoder_outputs += [_decoder_outputs]
            rnn_outputs += [_rnn_outputs]
            copy_attentions += [_copy_attentions]
            coref_attentions += [_coref_attentions]
            predictions += [_predictions]
            coref_indexes += [corefs]
            decoder_mask += [_mask]
        decoder_input_history = torch.cat(decoder_input_history[1:], dim=1)
        decoder_outputs = torch.cat(decoder_outputs[1:], dim=1)
        rnn_outputs = torch.cat(rnn_outputs[1:], dim=1)
        predictions = torch.cat(predictions[:-1], dim=1)
        coref_indexes = torch.cat(coref_indexes[:-1], dim=1)
        decoder_mask = 1 - torch.cat(decoder_mask[:-1], dim=1)
        return dict(predictions=predictions, coref_indexes=coref_indexes, decoder_mask=decoder_mask, decoder_inputs=decoder_input_history, decoder_memory_bank=decoder_outputs, decoder_rnn_memory_bank=rnn_outputs, copy_attentions=copy_attentions, coref_attentions=coref_attentions)

    def _update_maps_and_get_next_input(self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps, copy_vocabs, masks, tag_luts, invalid_indexes):
        """Dynamically update/build the maps needed for copying.

        :param step: the decoding step, int.
        :param predictions: [batch_size]
        :param copy_vocab_size: int.
        :param coref_attention_maps: [batch_size, max_decode_length, max_decode_length]
        :param coref_vocab_maps:  [batch_size, max_decode_length]
        :param copy_vocabs: a list of dynamic vocabs.
        :param masks: a list of [batch_size] tensors indicating whether EOS has been generated.
            if EOS has has been generated, then the mask is `1`.
        :param tag_luts: a dict mapping key to a list of dicts mapping a source token to a POS tag.
        :param invalid_indexes: a dict storing invalid indexes for copying and generation.
        :return:
        """
        vocab_size = self.generator.vocab_size
        batch_size = predictions.size(0)
        batch_index = torch.arange(0, batch_size).type_as(predictions)
        step_index = torch.full_like(predictions, step)
        gen_mask = predictions.lt(vocab_size)
        copy_mask = predictions.ge(vocab_size).mul(predictions.lt(vocab_size + copy_vocab_size))
        coref_mask = predictions.ge(vocab_size + copy_vocab_size)
        coref_index = predictions - vocab_size - copy_vocab_size
        coref_index.masked_fill_(1 - coref_mask, step + 1)
        coref_attention_maps[batch_index, step_index, coref_index] = 1
        coref_predictions = (predictions - vocab_size - copy_vocab_size) * coref_mask.long()
        coref_predictions = coref_vocab_maps.gather(1, coref_predictions.unsqueeze(1)).squeeze(1)
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        pos_tags = torch.full_like(predictions, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
        for i, index in enumerate(copy_predictions.tolist()):
            copied_token = copy_vocabs[i].get_token_from_idx(index)
            if index != 0:
                pos_tags[i] = self.vocab.get_token_index(tag_luts[i]['pos'][copied_token], 'pos_tags')
                if False:
                    invalid_indexes['source_copy'][i].add(index)
            copy_predictions[i] = self.vocab.get_token_index(copied_token, 'decoder_token_ids')
        for i, index in enumerate((predictions * gen_mask.long() + coref_predictions * coref_mask.long()).tolist()):
            if index != 0:
                token = self.vocab.get_token_from_index(index, 'decoder_token_ids')
                src_token = find_similar_token(token, list(tag_luts[i]['pos'].keys()))
                if src_token is not None:
                    pos_tags[i] = self.vocab.get_token_index(tag_luts[i]['pos'][src_token], 'pos_tag')
                if False:
                    invalid_indexes['vocab'][i].add(index)
        next_input = coref_predictions * coref_mask.long() + copy_predictions * copy_mask.long() + predictions * gen_mask.long()
        coref_vocab_maps[batch_index, step_index + 1] = next_input
        coref_resolved_preds = coref_predictions * coref_mask.long() + predictions * (1 - coref_mask).long()
        has_eos = torch.zeros_like(gen_mask)
        if len(masks) != 0:
            has_eos = torch.cat(masks, 1).long().sum(1).gt(0)
        mask = next_input.eq(self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')) | has_eos
        return next_input.unsqueeze(1), coref_resolved_preds.unsqueeze(1), pos_tags.unsqueeze(1), coref_index.unsqueeze(1), mask.unsqueeze(1)

    def decode_with_graph_parser(self, decoder_inputs, memory_bank, corefs, mask):
        """Predict edges and edge labels between nodes.
        :param decoder_inputs: [batch_size, node_length, embedding_size]
        :param memory_bank: [batch_size, node_length, hidden_size]
        :param corefs: [batch_size, node_length]
        :param mask:  [batch_size, node_length]
        :return a dict of edge_heads and edge_labels.
            edge_heads: [batch_size, node_length]
            edge_labels: [batch_size, node_length]
        """
        if self.use_aux_encoder:
            aux_encoder_outputs = self.aux_encoder(decoder_inputs, mask)
            self.aux_encoder.reset_states()
            memory_bank = torch.cat([memory_bank, aux_encoder_outputs], 2)
        memory_bank, _, _, corefs, mask = self.graph_decoder._add_head_sentinel(memory_bank, None, None, corefs, mask)
        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.graph_decoder.encode(memory_bank)
        edge_node_scores = self.graph_decoder._get_edge_node_scores(edge_node_h, edge_node_m, mask.float())
        edge_heads, edge_labels = self.graph_decoder.mst_decode(edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        return dict(edge_heads=edge_heads, edge_labels=edge_labels)

    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building the STOG Model...')
        encoder_input_size = 0
        bert_encoder = None
        if params.get('use_bert', False):
            bert_encoder = Seq2SeqBertEncoder.from_pretrained(params['bert']['pretrained_model_dir'])
            encoder_input_size += params['bert']['hidden_size']
            for p in bert_encoder.parameters():
                p.requires_grad = False
        encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        encoder_input_size += params['encoder_token_embedding']['embedding_dim']
        encoder_pos_embedding = Embedding.from_params(vocab, params['encoder_pos_embedding'])
        encoder_input_size += params['encoder_pos_embedding']['embedding_dim']
        encoder_must_copy_embedding = None
        if params.get('use_must_copy_embedding', False):
            encoder_must_copy_embedding = Embedding.from_params(vocab, params['encoder_must_copy_embedding'])
            encoder_input_size += params['encoder_must_copy_embedding']['embedding_dim']
        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
            encoder_char_cnn = CnnEncoder(embedding_dim=params['encoder_char_cnn']['embedding_dim'], num_filters=params['encoder_char_cnn']['num_filters'], ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'], conv_layer_activation=torch.tanh)
            encoder_input_size += params['encoder_char_cnn']['num_filters']
        else:
            encoder_char_embedding = None
            encoder_char_cnn = None
        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])
        params['encoder']['input_size'] = encoder_input_size
        encoder = PytorchSeq2SeqWrapper(module=StackedBidirectionalLstm.from_params(params['encoder']), stateful=True)
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])
        decoder_input_size = params['decoder']['hidden_size']
        decoder_input_size += params['decoder_token_embedding']['embedding_dim']
        decoder_input_size += params['decoder_coref_embedding']['embedding_dim']
        decoder_input_size += params['decoder_pos_embedding']['embedding_dim']
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        decoder_coref_embedding = Embedding.from_params(vocab, params['decoder_coref_embedding'])
        decoder_pos_embedding = Embedding.from_params(vocab, params['decoder_pos_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(embedding_dim=params['decoder_char_cnn']['embedding_dim'], num_filters=params['decoder_char_cnn']['num_filters'], ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'], conv_layer_activation=torch.tanh)
            decoder_input_size += params['decoder_char_cnn']['num_filters']
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None
        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])
        if params['source_attention']['attention_function'] == 'mlp':
            source_attention = MLPAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['encoder']['hidden_size'] * 2, attention_hidden_size=params['decoder']['hidden_size'], coverage=params['source_attention'].get('coverage', False))
        else:
            source_attention = DotProductAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['encoder']['hidden_size'] * 2, share_linear=params['source_attention'].get('share_linear', False))
        source_attention_layer = GlobalAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['encoder']['hidden_size'] * 2, attention=source_attention)
        if params['coref_attention']['attention_function'] == 'mlp':
            coref_attention = MLPAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['decoder']['hidden_size'], attention_hidden_size=params['decoder']['hidden_size'], coverage=params['coref_attention'].get('coverage', False), use_concat=params['coref_attention'].get('use_concat', False))
        elif params['coref_attention']['attention_function'] == 'biaffine':
            coref_attention = BiaffineAttention(input_size_decoder=params['decoder']['hidden_size'], input_size_encoder=params['encoder']['hidden_size'] * 2, hidden_size=params['coref_attention']['hidden_size'])
        else:
            coref_attention = DotProductAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['decoder']['hidden_size'], share_linear=params['coref_attention'].get('share_linear', True))
        coref_attention_layer = GlobalAttention(decoder_hidden_size=params['decoder']['hidden_size'], encoder_hidden_size=params['decoder']['hidden_size'], attention=coref_attention)
        params['decoder']['input_size'] = decoder_input_size
        decoder = InputFeedRNNDecoder(rnn_cell=StackedLstm.from_params(params['decoder']), attention_layer=source_attention_layer, coref_attention_layer=coref_attention_layer, dropout=InputVariationalDropout(p=params['decoder']['dropout']), use_coverage=params['use_coverage'])
        if params.get('use_aux_encoder', False):
            aux_encoder = PytorchSeq2SeqWrapper(module=StackedBidirectionalLstm.from_params(params['aux_encoder']), stateful=True)
            aux_encoder_output_dropout = InputVariationalDropout(p=params['aux_encoder']['dropout'])
        else:
            aux_encoder = None
            aux_encoder_output_dropout = None
        switch_input_size = params['encoder']['hidden_size'] * 2
        generator = PointerGenerator(input_size=params['decoder']['hidden_size'], switch_input_size=switch_input_size, vocab_size=vocab.get_vocab_size('decoder_token_ids'), force_copy=params['generator'].get('force_copy', True), vocab_pad_idx=0)
        graph_decoder = DeepBiaffineGraphDecoder.from_params(vocab, params['graph_decoder'])
        punctuation_ids = []
        oov_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'decoder_token_ids')
        for c in ',.?!:;"\'-(){}[]':
            c_id = vocab.get_token_index(c, 'decoder_token_ids')
            if c_id != oov_id:
                punctuation_ids.append(c_id)
        logger.info('encoder_token: %d' % vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' % vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' % vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' % vocab.get_vocab_size('decoder_token_characters'))
        return cls(vocab=vocab, punctuation_ids=punctuation_ids, use_must_copy_embedding=params.get('use_must_copy_embedding', False), use_char_cnn=params['use_char_cnn'], use_coverage=params['use_coverage'], use_aux_encoder=params.get('use_aux_encoder', False), use_bert=params.get('use_bert', False), max_decode_length=params.get('max_decode_length', 50), bert_encoder=bert_encoder, encoder_token_embedding=encoder_token_embedding, encoder_pos_embedding=encoder_pos_embedding, encoder_must_copy_embedding=encoder_must_copy_embedding, encoder_char_embedding=encoder_char_embedding, encoder_char_cnn=encoder_char_cnn, encoder_embedding_dropout=encoder_embedding_dropout, encoder=encoder, encoder_output_dropout=encoder_output_dropout, decoder_token_embedding=decoder_token_embedding, decoder_coref_embedding=decoder_coref_embedding, decoder_pos_embedding=decoder_pos_embedding, decoder_char_cnn=decoder_char_cnn, decoder_char_embedding=decoder_char_embedding, decoder_embedding_dropout=decoder_embedding_dropout, decoder=decoder, aux_encoder=aux_encoder, aux_encoder_output_dropout=aux_encoder_output_dropout, generator=generator, graph_decoder=graph_decoder, test_config=params.get('mimick_test', None))


class Generator(torch.nn.Module):

    def __init__(self, input_size, vocab_size, pad_idx):
        super(Generator, self).__init__()
        self._generator = torch.nn.Sequential(torch.nn.Linear(input_size, vocab_size), torch.nn.LogSoftmax(dim=-1))
        self.criterion = torch.nn.NLLLoss(ignore_index=pad_idx, reduction='sum')
        self.metrics = Seq2SeqMetrics()
        self.pad_idx = pad_idx

    def forward(self, inputs):
        """Transform inputs to vocab-size space and compute logits.

        :param inputs:  [batch, seq_length, input_size]
        :return:  [batch, seq_length, vocab_size]
        """
        batch_size, seq_length, _ = inputs.size()
        inputs = inputs.view(batch_size * seq_length, -1)
        scores = self._generator(inputs)
        scores = scores.view(batch_size, seq_length, -1)
        _, predictions = scores.max(2)
        return dict(scores=scores, predictions=predictions)

    def compute_loss(self, inputs, targets):
        batch_size, seq_length, _ = inputs.size()
        output = self(inputs)
        scores = output['scores'].view(batch_size * seq_length, -1)
        predictions = output['predictions'].view(-1)
        targets = targets.view(-1)
        loss = self.criterion(scores, targets)
        non_pad = targets.ne(self.pad_idx)
        num_correct = predictions.eq(targets).masked_select(non_pad).sum().item()
        num_non_pad = non_pad.sum().item()
        self.metrics(loss.item(), num_non_pad, num_correct)
        return dict(loss=loss.div(float(num_non_pad)), predictions=output['predictions'])

    @classmethod
    def from_params(cls, params):
        return cls(input_size=params['input_size'], vocab_size=params['vocab_size'], pad_idx=params['pad_idx'])


class TextFieldEmbedder(torch.nn.Module, Registrable):
    """
    A ``TextFieldEmbedder`` is a ``Module`` that takes as input the
    :class:`~allennlp.data.DataArray` produced by a :class:`~allennlp.data.fields.TextField` and
    returns as output an embedded representation of the tokens in that field.

    The ``DataArrays`` produced by ``TextFields`` are `dictionaries` with named representations,
    like "words" and "characters".  When you create a ``TextField``, you pass in a dictionary of
    :class:`~allennlp.data.TokenIndexer` objects, telling the field how exactly the tokens in the
    field should be represented.  This class changes the type signature of ``Module.forward``,
    restricting ``TextFieldEmbedders`` to take inputs corresponding to a single ``TextField``,
    which is a dictionary of tensors with the same names as were passed to the ``TextField``.

    We also add a method to the basic ``Module`` API: :func:`get_output_dim()`.  You might need
    this if you want to construct a ``Linear`` layer using the output of this embedder, for
    instance.
    """
    default_implementation = 'basic'

    def forward(self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int=0) ->torch.Tensor:
        """
        Parameters
        ----------
        text_field_input : ``Dict[str, torch.Tensor]``
            A dictionary that was the output of a call to ``TextField.as_tensor``.  Each tensor in
            here is assumed to have a shape roughly similar to ``(batch_size, sequence_length)``
            (perhaps with an extra trailing dimension for the characters in each token).
        num_wrapping_dims : ``int``, optional (default=0)
            If you have a ``ListField[TextField]`` that created the ``text_field_input``, you'll
            end up with tensors of shape ``(batch_size, wrapping_dim1, wrapping_dim2, ...,
            sequence_length)``.  This parameter tells us how many wrapping dimensions there are, so
            that we can correctly ``TimeDistribute`` the embedding of each named representation.
        """
        raise NotImplementedError

    def get_output_dim(self) ->int:
        """
        Returns the dimension of the vector representing each token in the output of this
        ``TextFieldEmbedder``.  This is `not` the shape of the returned tensor, but the last element of
        that shape.
        """
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiLinear,
     lambda: ([], {'left_features': 4, 'right_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BiaffineAttention,
     lambda: ([], {'input_size_encoder': 4, 'input_size_decoder': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (DotProductAttention,
     lambda: ([], {'decoder_hidden_size': 4, 'encoder_hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (InputVariationalDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPAttention,
     lambda: ([], {'decoder_hidden_size': 4, 'encoder_hidden_size': 4, 'attention_hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_sheng_z_stog(_paritybench_base):
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

