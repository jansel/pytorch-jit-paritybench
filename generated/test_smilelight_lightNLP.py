import sys
_module = sys.modules[__name__]
del sys
test_cb = _module
test_cbow = _module
test_cws = _module
test_gdp = _module
test_lm = _module
test_mt = _module
test_ner = _module
test_pos = _module
test_re = _module
test_sa = _module
test_skip_gram = _module
test_srl = _module
test_ss = _module
test_tdp = _module
test_te = _module
test_ts = _module
test_word_vectors = _module
lightnlp = _module
base = _module
config = _module
model = _module
module = _module
tool = _module
sl = _module
cws = _module
model = _module
module = _module
tool = _module
utils = _module
convert = _module
ner = _module
model = _module
module = _module
tool = _module
pos = _module
model = _module
module = _module
tool = _module
srl = _module
model = _module
module = _module
tool = _module
sp = _module
gdp = _module
components = _module
biaffine = _module
dropout = _module
lstm = _module
mlp = _module
model = _module
module = _module
tool = _module
dataset = _module
metric = _module
reader = _module
vocab = _module
tdp = _module
action_chooser = _module
combiner = _module
word_embedding = _module
model = _module
module = _module
tool = _module
feature_extractor = _module
parser_state = _module
vectors = _module
sr = _module
ss = _module
model = _module
module = _module
tool = _module
pad = _module
te = _module
model = _module
module = _module
tool = _module
tc = _module
re = _module
model = _module
module = _module
tool = _module
preprocess = _module
sa = _module
model = _module
module = _module
tool = _module
tg = _module
cb = _module
models = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
tool = _module
lm = _module
model = _module
module = _module
tool = _module
mt = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
tool = _module
ts = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
tool = _module
deploy = _module
learning = _module
log = _module
score_func = _module
word_vector = _module
we = _module
cbow = _module
model = _module
module = _module
hierarchical_softmax = _module
model = _module
module = _module
model = _module
negative_sampling = _module
model = _module
module = _module
tool = _module
huffman_tree = _module
sampling = _module
skip_gram = _module
model = _module
module = _module
model = _module
module = _module
model = _module
model = _module
module = _module
tool = _module
sampling = _module
setup = _module
test_flask = _module

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


from torchtext.vocab import Vectors


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.utils.tensorboard import SummaryWriter


from torchtext.data import Dataset


from torchtext.data import Field


from torchtext.data import BucketIterator


from torchtext.data import ReversibleField


from torchtext.datasets import SequenceTaggingDataset


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


from sklearn.metrics import recall_score


from sklearn.metrics import precision_score


from torch.nn.utils.rnn import PackedSequence


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


from collections import namedtuple


from collections import Counter


from collections import deque


import torch.autograd as ag


import re


from torchtext.data import TabularDataset


from torchtext.data import Iterator


import random


from torch.nn.utils import clip_grad_norm_


from torchtext.data import BPTTIterator


from torchtext.datasets import LanguageModelingDataset


from typing import List


from torchtext.vocab import Vocab


LEVEL_COLOR = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {'black': '30', 'red': '31', 'green': '32', 'yellow': '33', 'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37', 'bg_black': '40', 'bg_red': '41', 'bg_green': '42', 'bg_yellow': '43', 'bg_blue': '44', 'bg_magenta': '45', 'bg_cyan': '46', 'bg_white': '47', 'light_black': '1;30', 'light_red': '1;31', 'light_green': '1;32', 'light_yellow': '1;33', 'light_blue': '1;34', 'light_magenta': '1;35', 'light_cyan': '1;36', 'light_white': '1;37', 'light_bg_black': '100', 'light_bg_red': '101', 'light_bg_green': '102', 'light_bg_yellow': '103', 'light_bg_blue': '104', 'light_bg_magenta': '105', 'light_bg_cyan': '106', 'light_bg_white': '107'}

    def __init__(self, fmt, datefmt):
        super(ColoredFormatter, self).__init__(fmt, datefmt)

    def parse_color(self, level_name):
        color_name = LEVEL_COLOR.get(level_name, '')
        if not color_name:
            return ''
        color_value = []
        color_name = color_name.split(',')
        for _cn in color_name:
            color_code = self.COLOR_MAP.get(_cn, '')
            if color_code:
                color_value.append(color_code)
        return '\x1b[' + ';'.join(color_value) + 'm'

    def format(self, record):
        record.log_color = self.parse_color(record.levelname)
        message = super(ColoredFormatter, self).format(record) + '\x1b[0m'
        return message


FILE_DATE_FMT = '%Y-%m-%d %H:%M:%S'


FILE_LOG_FMT = '[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'


STDOUT_DATE_FMT = '%Y-%m-%d %H:%M:%S'


STDOUT_LOG_FMT = '%(log_color)s[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'


def _get_logger(log_to_file=True, log_filename='default.log', log_level='DEBUG'):
    _logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(ColoredFormatter(fmt=STDOUT_LOG_FMT, datefmt=STDOUT_DATE_FMT))
    _logger.addHandler(stdout_handler)
    if log_to_file:
        _tmp_path = os.path.dirname(os.path.abspath(__file__))
        _tmp_path = os.path.join(_tmp_path, '../logs/{}'.format(log_filename))
        file_handler = logging.handlers.TimedRotatingFileHandler(_tmp_path, when='midnight', backupCount=30)
        file_formatter = logging.Formatter(fmt=FILE_LOG_FMT, datefmt=FILE_DATE_FMT)
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)
    _logger.setLevel(log_level)
    return _logger


logger = _get_logger(log_to_file=False)


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loadding model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLstmCrf(BaseModel):

    def __init__(self, args):
        super(BiLstmCrf, self).__init__(args)
        self.args = args
        self.hidden_dim = 300
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        pos_size = args.pos_size
        pos_dim = args.pos_dim
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.word_embedding = nn.Embedding.from_pretrained(vectors, freeze=not args.non_static)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim)
        self.lstm = nn.LSTM(embedding_dimension + pos_dim + 1, self.hidden_dim // 2, bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num)
        self.crflayer = CRF(self.tag_num)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def loss(self, x, sent_lengths, pos, rel, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, pos, rel, sent_lengths)
        return self.crflayer(emissions, y, mask=mask)

    def forward(self, x, poses, rels, sent_lengths):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, poses, rels, sent_lengths)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, poses, rels, sent_lengths):
        word = self.word_embedding(sentence.to(DEVICE))
        pos = self.pos_embedding(poses.to(DEVICE))
        rels = rels.view(rels.size(0), rels.size(1), 1).float()
        x = torch.cat((word, pos, rels), dim=2)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size)
        y = self.hidden2label(lstm_out)
        return y


DEFAULT_CONFIG = {'save_path': './saves'}


class BaseConfig(object):

    def __init__(self):
        pass

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path']):
        config_path = os.path.join(path, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logger.info('loadding config from {}'.format(config_path))
        config.save_path = path
        return config

    def save(self, path=None):
        if not hasattr(self, 'save_path'):
            raise AttributeError('config object must init save_path attr in init method!')
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, 'config.pkl')
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved config to {}'.format(config_path))


class Config(BaseConfig):

    def __init__(self, word_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def bis_cws(words, tags):
    assert len(words) == len(tags)
    poses = []
    for i, tag in enumerate(tags):
        if tag in ['B', 'S']:
            begin = i
        if i == len(tags) - 1:
            poses.append(''.join(words[begin:i + 1]))
        elif tags[i + 1] != 'I':
            poses.append(''.join(words[begin:i + 1]))
            begin = i + 1
    return poses


WORD = Field(tokenize=lambda x: [x], batch_first=True)


Fields = [('context', WORD), ('target', WORD)]


def light_tokenize(text):
    return [word for word in jieba.cut(text) if word.strip()]


TAG = Field(sequential=True, tokenize=light_tokenize, is_target=True, unk_token=None)


TEXT = Field(lower=True, tokenize=light_tokenize, include_lengths=True, batch_first=True, init_token='<sos>', eos_token='<eos>')


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


class CWS(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = cws_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = cws_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = cws_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = cws_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        train_iter = cws_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = -bilstmcrf.loss(item_text_sentences, item_text_lengths, item.tag) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('cws_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('cws_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        bilstmcrf.save()

    def predict(self, text):
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)])
        vec_predict = self._model(vec_text.view(-1, 1), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return bis_cws([x for x in text], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab

    def test(self, test_path):
        test_dataset = cws_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = cws_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab, self._tag_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def deploy(self, route_path='/cws', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            text = request.args.get('text', '')
            result = self.predict(text)
            return flask.jsonify({'state': 'OK', 'result': {'words': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('_')[0] == 'O':
            ranges.append({'entity': ''.join(words[begin:i + 1]), 'type': temp_type, 'start': begin, 'end': i})
    for i, tag in enumerate(tags):
        if tag.split('_')[0] == 'O':
            pass
        elif tag.split('_')[0] == 'B':
            begin = i
            temp_type = tag.split('_')[1]
            check_if_closing_range()
        elif tag.split('_')[0] == 'I':
            check_if_closing_range()
    return ranges


class NER(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = ner_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ner_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        train_iter = ner_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = -bilstmcrf.loss(item_text_sentences, item_text_lengths, item.tag) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('ner_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('ner_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        bilstmcrf.save()

    def predict(self, text):
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)])
        vec_predict = self._model(vec_text.view(-1, 1), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return iob_ranges([x for x in text], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab

    def test(self, test_path):
        test_dataset = ner_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = ner_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab, self._tag_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def deploy(self, route_path='/ner', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            text = request.args.get('text', '')
            result = self.predict(text)
            return flask.jsonify({'state': 'OK', 'result': {'result': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


ROOT = '<ROOT>'


POS = Field(sequential=True, tokenize=light_tokenize, unk_token=None, batch_first=True, init_token=ROOT)


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'rel':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iobes_ranges(words, tags):
    new_tags = iobes_iob(tags)
    return iob_ranges(words, new_tags)


class SRL(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None
        self._pos_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = srl_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = srl_tool.get_dataset(dev_path)
            word_vocab, pos_vocab, tag_vocab = srl_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, pos_vocab, tag_vocab = srl_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._pos_vocab = pos_vocab
        self._tag_vocab = tag_vocab
        train_iter = srl_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, pos_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = -bilstmcrf.loss(item_text_sentences, item_text_lengths, item.pos, item.rel, item.tag) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('srl_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('srl_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        bilstmcrf.save()

    def predict(self, word_list, pos_list, rel_list):
        self._model.eval()
        assert len(word_list) == len(pos_list) == len(rel_list)
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in word_list]).view(-1, 1)
        len_text = torch.tensor([len(vec_text)])
        vec_pos = torch.tensor([self._pos_vocab.stoi[x] for x in pos_list]).view(-1, 1)
        vec_rel = torch.tensor([int(x) for x in rel_list]).view(-1, 1)
        vec_predict = self._model(vec_text, vec_pos, vec_rel, len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return iobes_ranges([x for x in word_list], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
        self._pos_vocab = config.pos_vocab

    def test(self, test_path):
        test_dataset = srl_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = srl_tool.get_score(self._model, dev_item.text, dev_item.tag, dev_item.pos, dev_item.rel, self._word_vocab, self._tag_vocab, self._pos_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def deploy(self, route_path='/srl', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            word_list = request.args.get('word_list', [])
            pos_list = request.args.get('pos_list', [])
            rel_list = request.args.get('rel_list', [])
            result = self.predict(word_list, pos_list, rel_list)
            return flask.jsonify({'state': 'OK', 'result': {'result': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        info = f'n_in={self.n_in}, n_out={self.n_out}'
        if self.bias_x:
            info += f', bias_x={self.bias_x}'
        if self.bias_y:
            info += f', bias_y={self.bias_y}'
        return info

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        s = x @ self.weight @ torch.transpose(y, -1, -2)
        s = s.squeeze(1)
        return s


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()
        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        info = f'p={self.p}'
        if self.batch_first:
            info += f', batch_first={self.batch_first}'
        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, (0)], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask
        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)
        return mask


class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def forward(self, x, y, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))
            scale = 3.0 / (2.0 * x_mask + y_mask + eps)
            x_mask *= scale
            y_mask *= scale
            x *= x_mask.unsqueeze(dim=-1)
            y *= y_mask.unsqueeze(dim=-1)
        return x, y


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.dropout)
        for t in steps:
            batch_size = batch_sizes[t]
            if len(h) < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
            last_batch_size = batch_size
        if reverse:
            output.reverse()
        output = torch.cat(output)
        return output

    def forward(self, x, hx=None):
        x, batch_sizes = x
        batch_size = batch_sizes[0]
        if hx is None:
            init = x.new_zeros(batch_size, self.hidden_size)
            hx = init, init
        for layer in range(self.num_layers):
            if self.training:
                mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
                mask = torch.cat([mask[:batch_size] for batch_size in batch_sizes])
                x *= mask
            x = torch.split(x, batch_sizes.tolist())
            f_output = self.layer_forward(x=x, hx=hx, cell=self.f_cells[layer], batch_sizes=batch_sizes, reverse=False)
            if self.bidirectional:
                b_output = self.layer_forward(x=x, hx=hx, cell=self.b_cells[layer], batch_sizes=batch_sizes, reverse=True)
            if self.bidirectional:
                x = torch.cat([f_output, b_output], -1)
            else:
                x = f_output
        x = PackedSequence(x, batch_sizes)
        return x


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout):
        super(MLP, self).__init__()
        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BiaffineParser(BaseModel):

    def __init__(self, args):
        super(BiaffineParser, self).__init__(args)
        self.args = args
        self.hidden_dim = args.lstm_hidden
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.lstm_layters = args.lstm_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        word_dim = args.word_dim
        pos_num = args.pos_num
        pos_dim = args.pos_dim
        self.word_embedding = nn.Embedding(vocabulary_size, word_dim)
        vectors = Vectors(args.vector_path).vectors
        self.pretrained_embedding = nn.Embedding.from_pretrained(vectors)
        self.pos_embedding = nn.Embedding(pos_num, pos_dim)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)
        self.lstm = LSTM(word_dim + pos_dim, self.hidden_dim, bidirectional=self.bidirectional, num_layers=self.lstm_layters, dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        self.mlp_arc_h = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout)
        self.arc_attn = Biaffine(n_in=args.mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=args.mlp_rel, n_out=args.ref_num, bias_x=True, bias_y=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embedding.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def forward(self, words, tags):
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        embed = self.pretrained_embedding(words)
        embed += self.word_embedding(words.masked_fill_(words.ge(self.word_embedding.num_embeddings), 0))
        tag_embed = self.pos_embedding(tags)
        embed, tag_embed = self.embed_dropout(embed, tag_embed)
        x = torch.cat((embed, tag_embed), dim=-1)
        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        s_arc.masked_fill_((1 - mask).unsqueeze(1), float('-inf'))
        return s_arc, s_rel


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AttachmentMethod(Metric):

    def __init__(self, eps=1e-05):
        super(AttachmentMethod, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __call__(self, pred_arcs, pred_rels, gold_arcs, gold_rels):
        arc_mask = pred_arcs.eq(gold_arcs)
        rel_mask = pred_rels.eq(gold_rels) & arc_mask
        self.total += len(arc_mask)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_rels += rel_mask.sum().item()

    def __repr__(self):
        return f'UAS: {self.uas:.2%} LAS: {self.las:.2%}'

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


REF = Field(sequential=True, tokenize=light_tokenize, unk_token=None, batch_first=True, init_token=ROOT)


def post_process(arr, _):
    return [[int(item) for item in arr_item] for arr_item in arr]


HEAD = Field(sequential=True, use_vocab=False, unk_token=None, pad_token=0, postprocessing=post_process, batch_first=True, init_token=0)


class GDP(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._pos_vocab = None
        self._ref_vocab = None
        self._pad_index = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = gdp_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = gdp_tool.get_dataset(dev_path)
            word_vocab, pos_vocab, head_vocab, ref_vocab = gdp_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, pos_vocab, head_vocab, ref_vocab = gdp_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._pos_vocab = pos_vocab
        self._ref_vocab = ref_vocab
        train_iter = gdp_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, pos_vocab, ref_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        biaffine_parser = BiaffineParser(config)
        self._model = biaffine_parser
        self._pad_index = config.pad_index
        optim = torch.optim.Adam(biaffine_parser.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2), eps=config.epsilon)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda x: config.decay ** (x / config.decay_steps))
        for epoch in range(config.epoch):
            biaffine_parser.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                biaffine_parser.zero_grad()
                words = item.word
                tags = item.pos
                refs = item.ref
                arcs = item.head
                mask = words.ne(config.pad_index)
                mask[:, (0)] = 0
                s_arc, s_rel = self._model(words, tags)
                s_arc, s_rel = s_arc[mask], s_rel[mask]
                gold_arcs, gold_rels = arcs[mask], refs[mask]
                item_loss = self._get_loss(s_arc, s_rel, gold_arcs, gold_rels)
                acc_loss += item_loss.cpu().item()
                item_loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optim.step()
                scheduler.step()
            acc_loss /= len(train_iter)
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('gdp_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score, dev_metric = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                logger.info('metric:{}'.format(dev_metric))
                writer.add_scalar('gdp_train/dev_score', dev_score, epoch)
            writer.flush()
        writer.close()
        config.save()
        biaffine_parser.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        biaffine_parser = BiaffineParser(config)
        biaffine_parser.load()
        self._model = biaffine_parser
        self._word_vocab = config.word_vocab
        self._pos_vocab = config.pos_vocab
        self._ref_vocab = config.ref_vocab
        self._pad_index = config.pad_index
        self._check_vocab()

    def test(self, test_path):
        test_dataset = gdp_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _check_vocab(self):
        if not hasattr(WORD, 'vocab'):
            WORD.vocab = self._word_vocab
        if not hasattr(POS, 'vocab'):
            POS.vocab = self._pos_vocab
        if not hasattr(REF, 'vocab'):
            REF.vocab = self._ref_vocab

    def _validate(self, dev_dataset):
        self._model.eval()
        loss, metric = 0, AttachmentMethod()
        dev_iter = gdp_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            mask = dev_item.word.ne(self._pad_index)
            mask[:, (0)] = 0
            s_arc, s_rel = self._model(dev_item.word, dev_item.pos)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = dev_item.head[mask], dev_item.ref[mask]
            pred_arcs, pred_rels = self._decode(s_arc, s_rel)
            loss += self._get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(dev_iter)
        return loss, metric

    def predict(self, word_list: list, pos_list: list):
        self._model.eval()
        assert len(word_list) == len(pos_list)
        word_list.insert(0, ROOT)
        pos_list.insert(0, ROOT)
        vec_word = WORD.numericalize([word_list])
        vec_pos = POS.numericalize([pos_list])
        mask = vec_word.ne(self._pad_index)
        s_arc, s_rel = self._model(vec_word, vec_pos)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        pred_arcs, pred_rels = self._decode(s_arc, s_rel)
        pred_arcs = pred_arcs.cpu().tolist()
        pred_arcs[0] = 0
        pred_rels = [self._ref_vocab.itos[rel] for rel in pred_rels]
        pred_rels[0] = ROOT
        return pred_arcs, pred_rels

    def _get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        arc_loss = F.cross_entropy(s_arc, gold_arcs)
        rel_loss = F.cross_entropy(s_rel, gold_rels)
        loss = arc_loss + rel_loss
        return loss

    def _decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        return pred_arcs, pred_rels

    def deploy(self, route_path='/cb', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            words = request.args.get('words', '')
            pos = request.args.get('pos', '')
            result = self.predict(words, pos)
            return flask.jsonify({'state': 'OK', 'result': {'arcs': result[0], 'rels': result[1]}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class Actions:
    """Simple Enum for each possible parser action"""
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2
    NUM_ACTIONS = 3
    action_to_ix = {'SHIFT': SHIFT, 'REDUCE_L': REDUCE_L, 'REDUCE_R': REDUCE_R}


class ActionChooserNetwork(nn.Module):

    def __init__(self, input_dim):
        super(ActionChooserNetwork, self).__init__()
        self.hidden_dim = input_dim
        self.linear1 = nn.Linear(input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, Actions.NUM_ACTIONS)

    def forward(self, inputs):
        input_vec = vectors.concat_and_flatten(inputs)
        temp_vec = self.linear1(input_vec)
        temp_vec = F.relu(temp_vec)
        result = self.linear2(temp_vec)
        return result


class MLPCombinerNetwork(nn.Module):

    def __init__(self, embedding_dim):
        super(MLPCombinerNetwork, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, head_embed, modifier_embed):
        input_vec = vectors.concat_and_flatten((head_embed, modifier_embed))
        temp_vec = self.linear1(input_vec)
        temp_vec = torch.tanh(temp_vec)
        result = self.linear2(temp_vec)
        return result


class LSTMCombinerNetwork(nn.Module):

    def __init__(self, embedding_dim, num_layers, dropout):
        super(LSTMCombinerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_cuda = False
        self.linear = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.hidden_dim = self.embedding_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, dropout=dropout)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, head_embed, modifier_embed):
        input_vec = vectors.concat_and_flatten((head_embed, modifier_embed))
        temp_vec = self.linear(input_vec).view(1, 1, -1)
        lstm_hiddens, self.hidden = self.lstm(temp_vec, self.hidden)
        return lstm_hiddens[-1]

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()


class VanillaWordEmbeddingLookup(nn.Module):
    """
    A component that simply returns a list of the word embeddings as
    autograd Variables.
    """

    def __init__(self, vocabulary_size, embedding_dim, vector_path=None, non_static=False):
        super(VanillaWordEmbeddingLookup, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(word_vectors, freeze=not non_static)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.to(DEVICE))
        return embeds


class BiLSTMWordEmbeddingLookup(nn.Module):

    def __init__(self, vocabulary_size, word_embedding_dim, hidden_dim, num_layers, dropout, vector_path=None, non_static=False):
        super(BiLSTMWordEmbeddingLookup, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.word_embedding_dim)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(word_vectors, freeze=not non_static)
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim // 2, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.hidden = self.init_hidden()

    def forward(self, sentence):
        embeddings = self.word_embeddings(sentence)
        lstm_hiddens, self.hidden = self.lstm(embeddings, self.hidden)
        return lstm_hiddens

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()


DepGraphEdge = namedtuple('DepGraphEdge', ['head', 'modifier'])


NULL_STACK_TOK = '<NULL-STACK>'


StackEntry = namedtuple('StackEntry', ['headword', 'headword_pos', 'embedding'])


class ParserState:

    def __init__(self, sentence, sentence_embs, combiner, null_stack_tok_embed=None):
        self.combiner = combiner
        self.curr_input_buff_idx = 0
        self.input_buffer = [StackEntry(we[0], pos, we[1]) for pos, we in enumerate(zip(sentence, sentence_embs))]
        self.stack = []
        self.null_stack_tok_embed = null_stack_tok_embed

    def shift(self):
        next_item = self.input_buffer[self.curr_input_buff_idx]
        self.stack.append(next_item)
        self.curr_input_buff_idx += 1

    def reduce_left(self):
        return self._reduce(Actions.REDUCE_L)

    def reduce_right(self):
        return self._reduce(Actions.REDUCE_R)

    def done_parsing(self):
        if len(self.stack) == 1 and self.curr_input_buff_idx == len(self.input_buffer) - 1:
            return True
        else:
            return False

    def stack_len(self):
        return len(self.stack)

    def input_buffer_len(self):
        return len(self.input_buffer) - self.curr_input_buff_idx

    def stack_peek_n(self, n):
        if len(self.stack) - n < 0:
            return [StackEntry(NULL_STACK_TOK, -1, self.null_stack_tok_embed)] * (n - len(self.stack)) + self.stack[:]
        return self.stack[-n:]

    def input_buffer_peek_n(self, n):
        assert self.curr_input_buff_idx + n - 1 <= len(self.input_buffer)
        return self.input_buffer[self.curr_input_buff_idx:self.curr_input_buff_idx + n]

    def _reduce(self, action):
        assert len(self.stack) >= 2, 'ERROR: Cannot reduce with stack length less than 2'
        if action == Actions.REDUCE_L:
            head = self.stack.pop()
            modifier = self.stack.pop()
        elif action == Actions.REDUCE_R:
            modifier = self.stack.pop()
            head = self.stack.pop()
        head_embedding = self.combiner(head.embedding, modifier.embedding)
        self.stack.append(StackEntry(head.headword, head.headword_pos, head_embedding))
        return DepGraphEdge((head.headword, head.headword_pos), (modifier.headword, modifier.headword_pos))

    def __str__(self):
        return 'Stack: {}\nInput Buffer: {}\n'.format([entry.headword for entry in self.stack], [entry.headword for entry in self.input_buffer[self.curr_input_buff_idx:]])


ROOT_TOK = '<ROOT>'


class SimpleFeatureExtractor:

    def get_features(self, parser_state, **kwargs):
        stack_len = 2
        input_buffer_len = 1
        stack_items = parser_state.stack_peek_n(stack_len)
        input_buffer_items = parser_state.input_buffer_peek_n(input_buffer_len)
        features = []
        assert len(stack_items) == stack_len
        assert len(input_buffer_items) == input_buffer_len
        features.extend([x.embedding for x in stack_items])
        features.extend([x.embedding for x in input_buffer_items])
        return features


class TransitionParser(BaseModel):

    def __init__(self, args):
        super(TransitionParser, self).__init__(args)
        self.args = args
        self.action_num = args.action_num
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.feature_extractor = SimpleFeatureExtractor() if args.feature_extractor == 'default' else None
        if args.embedding_type == 'lstm':
            self.word_embedding_component = BiLSTMWordEmbeddingLookup(vocabulary_size, args.word_embedding_dim, args.stack_embedding_dim, args.embedding_lstm_layers, args.embedding_lstm_dropout, args.vector_path, args.non_static)
        elif args.embedding_type == 'vanilla':
            self.word_embedding_component = VanillaWordEmbeddingLookup(vocabulary_size, args.word_embedding_dim)
        else:
            self.word_embedding_component = None
        self.action_chooser = ActionChooserNetwork(args.stack_embedding_dim * args.num_features) if args.action_chooser == 'default' else None
        if args.combiner == 'lstm':
            self.combiner = LSTMCombinerNetwork(args.stack_embedding_dim, args.combiner_lstm_layers, args.combiner_lstm_dropout)
        elif args.combiner == 'mlp':
            self.combiner = MLPCombinerNetwork(args.stack_embedding_dim)
        else:
            self.combiner = None
        self.null_stack_tok_embed = torch.randn(1, self.word_embedding_component.output_dim)

    def forward(self, sentence, actions=None):
        self.refresh()
        sentence_embs = self.word_embedding_component(sentence)
        parser_state = ParserState(sentence, sentence_embs, self.combiner, null_stack_tok_embed=self.null_stack_tok_embed)
        outputs = []
        actions_done = []
        dep_graph = set()
        if actions is not None:
            action_queue = deque()
            action_queue.extend([Actions.action_to_ix[a] for a in actions])
            have_gold_actions = True
        else:
            have_gold_actions = False
        while True:
            if parser_state.done_parsing():
                break
            features = self.feature_extractor.get_features(parser_state)
            log_probs = self.action_chooser(features)
            if have_gold_actions:
                temp_action = action_queue.popleft()
            else:
                temp_action = vectors.argmax(log_probs)
            if parser_state.input_buffer_len() == 1:
                temp_action = Actions.REDUCE_R
            elif parser_state.stack_len() < 2:
                temp_action = Actions.SHIFT
            if temp_action == Actions.SHIFT:
                parser_state.shift()
                reduction = None
            elif temp_action == Actions.REDUCE_L:
                reduction = parser_state.reduce_left()
            elif temp_action == Actions.REDUCE_R:
                reduction = parser_state.reduce_right()
            else:
                raise Exception('unvalid action!: {}'.format(temp_action))
            outputs.append(log_probs)
            if reduction:
                dep_graph.add(reduction)
            actions_done.append(temp_action)
        dep_graph.add(DepGraphEdge((TEXT.vocab.stoi[ROOT_TOK], -1), (parser_state.stack[-1].headword, parser_state.stack[-1].headword_pos)))
        return outputs, dep_graph, actions_done

    def refresh(self):
        if isinstance(self.combiner, LSTMCombinerNetwork):
            self.combiner.clear_hidden_state()
        if isinstance(self.word_embedding_component, BiLSTMWordEmbeddingLookup):
            self.word_embedding_component.clear_hidden_state()

    def predict(self, sentence):
        _, dep_graph, _ = self.forward(sentence)
        return dep_graph

    def predict_actions(self, sentence):
        _, _, actions_done = self.forward(sentence)
        return actions_done

    def to_cuda(self):
        self.word_embedding_component.use_cuda = True
        self.combiner.use_cuda = True
        self

    def to_cpu(self):
        self.word_embedding_component.use_cuda = False
        self.combiner.use_cuda = False
        self.cpu()


def action_tokenize(sequence: str):
    return [sequence]


ACTION = ReversibleField(sequential=True, tokenize=action_tokenize, is_target=True, unk_token=None, pad_token=None)


class TransitionDataset(Dataset):
    """Defines a Dataset of transition-based denpendency parsing format.
    eg:
    The bill intends ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_L
    The bill intends ||| SHIFT SHIFT REDUCE_L SHIFT REDUCE_L
    """

    def __init__(self, path, fields, encoding='utf-8', separator=' ||| ', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for inst in f:
                sentence, actions = inst.split(separator)
                sentence = sentence.strip().split()
                actions = actions.strip().split()
                examples.append(Example.fromlist((sentence, actions), fields))
        super(TransitionDataset, self).__init__(examples, fields, **kwargs)


class TDP(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._action_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = tdp_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = tdp_tool.get_dataset(dev_path)
            word_vocab, action_vocab = tdp_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, action_vocab = tdp_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._action_vocab = action_vocab
        train_iter = tdp_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, action_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        trainsition_parser = TransitionParser(config)
        self._model = trainsition_parser
        optim = torch.optim.Adam(trainsition_parser.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            trainsition_parser.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self._model.refresh()
                optim.zero_grad()
                outputs, dep_graph, actions_done = self._model(item.text)
                item_loss = 0
                for step_output, step_action in zip(outputs, item.action):
                    item_loss += F.cross_entropy(step_output, step_action)
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            acc_loss /= len(train_iter)
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('tdp_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('tdp_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        trainsition_parser.save()

    def predict(self, text):
        self._model.eval()
        sentences = light_tokenize(text)
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in sentences])
        outputs, dep_graph, actions_done = self._model(vec_text.view(-1, 1))
        results = set()
        for edge in dep_graph:
            results.add(DepGraphEdge((self._word_vocab.itos[edge.head[0]], edge.head[1]), (self._word_vocab.itos[edge.modifier[0]], edge.modifier[1])))
        return results

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        trainsition_parser = TransitionParser(config)
        trainsition_parser.load()
        self._model = trainsition_parser
        self._word_vocab = config.word_vocab
        self._action_vocab = config.action_vocab
        self._check_vocab()

    def test(self, test_path):
        test_dataset = tdp_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = tdp_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            self._model.refresh()
            item_score = tdp_tool.get_score(self._model, dev_item.text, dev_item.action)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def _check_vocab(self):
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        if not hasattr(ACTION, 'vocab'):
            ACTION.vocab = self._action_vocab

    def deploy(self, route_path='/tdp', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            text = request.args.get('text', '')
            result = self.predict(text)
            return flask.jsonify({'state': 'OK', 'result': {'results': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class MaLSTM(BaseModel):

    def __init__(self, args):
        super(MaLSTM, self).__init__(args)
        self.args = args
        self.hidden_dim = 300
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.pwd = torch.nn.PairwiseDistance(p=1)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def forward(self, left, right):
        left_vec = self.embedding(left.to(DEVICE))
        right_vec = self.embedding(right.to(DEVICE))
        self.hidden = self.init_hidden(batch_size=left.size(1))
        left_lstm_out, (left_lstm_hidden, _) = self.lstm(left_vec, self.hidden)
        right_lstm_out, (right_lstm_hidden, _) = self.lstm(right_vec, self.hidden)
        return self.manhattan_distance(left_lstm_hidden[0], right_lstm_hidden[0])

    def manhattan_distance(self, left, right):
        return torch.exp(-self.pwd(left, right))


LABEL = Field(sequential=False, unk_token=None)


def pad_sequnce(sequence, seq_length, pad_token='<pad>'):
    padded_seq = sequence[:]
    if len(padded_seq) < seq_length:
        padded_seq.extend([pad_token for _ in range(len(padded_seq), seq_length)])
    return padded_seq[:seq_length]


class SS(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._label_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = ss_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ss_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ss_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._label_vocab = tag_vocab
        train_iter = ss_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        malstm = MaLSTM(config)
        self._model = malstm
        optim = torch.optim.Adam(self._model.parameters(), lr=config.lr)
        loss_func = torch.nn.MSELoss()
        for epoch in range(config.epoch):
            self._model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self._model.zero_grad()
                left_text = item.texta
                right_text = item.textb
                predict_dis = self._model(left_text, right_text)
                item_loss = loss_func(predict_dis, item.label.type(torch.float32))
                acc_loss += item_loss.view(-1).cpu().item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('ss_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('ss_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        self._model.save()

    def predict(self, texta: str, textb: str):
        self._model.eval()
        pad_texta = pad_sequnce([x for x in texta], DEFAULT_CONFIG['fix_length'])
        vec_texta = torch.tensor([self._word_vocab.stoi[x] for x in pad_texta])
        pad_textb = pad_sequnce([x for x in textb], DEFAULT_CONFIG['fix_length'])
        vec_textb = torch.tensor([self._word_vocab.stoi[x] for x in pad_textb])
        vec_predict = self._model(vec_texta.view(-1, 1), vec_textb.view(-1, 1))[0]
        return vec_predict.cpu().item()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        malstm = MaLSTM(config)
        malstm.load()
        self._model = malstm
        self._word_vocab = config.word_vocab
        self._label_vocab = config.label_vocab

    def test(self, test_path):
        test_dataset = ss_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        if not hasattr(LABEL, 'vocab'):
            LABEL.vocab = self._label_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = ss_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            item_score = ss_tool.get_score(self._model, dev_item.texta, dev_item.textb, dev_item.label)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def deploy(self, route_path='/ss', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            texta = request.args.get('texta', '')
            textb = request.args.get('textb', '')
            result = self.predict(texta, textb)
            return flask.jsonify({'state': 'OK', 'result': {'prob': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class SharedLSTM(BaseModel):

    def __init__(self, args):
        super(SharedLSTM, self).__init__(args)
        self.args = args
        self.hidden_dim = 300
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.class_num)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def forward(self, left, right):
        left_vec = self.embedding(left.to(DEVICE))
        right_vec = self.embedding(right.to(DEVICE))
        self.hidden = self.init_hidden(batch_size=left.size(1))
        left_lstm_out, (left_lstm_hidden, _) = self.lstm(left_vec, self.hidden)
        right_lstm_out, (right_lstm_hidden, _) = self.lstm(right_vec, self.hidden)
        merged = torch.cat((left_lstm_out[-1], right_lstm_out[-1]), dim=1)
        merged = self.dropout_layer(merged)
        merged = self.batch_norm(merged)
        predict = self.hidden2label(merged)
        return predict


class TE(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._label_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = te_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = te_tool.get_dataset(dev_path)
            word_vocab, label_vocab = te_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, label_vocab = te_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._label_vocab = label_vocab
        train_iter = te_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, label_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        shared_lstm = SharedLSTM(config)
        self._model = shared_lstm
        optim = torch.optim.Adam(self._model.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            self._model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self._model.zero_grad()
                left_text = item.texta
                right_text = item.textb
                predict_dis = self._model(left_text, right_text)
                item_loss = F.cross_entropy(predict_dis, item.label)
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('te_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('te_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        self._model.save()

    def predict(self, texta: str, textb: str):
        self._model.eval()
        pad_texta = pad_sequnce([x for x in texta], DEFAULT_CONFIG['fix_length'])
        vec_texta = torch.tensor([self._word_vocab.stoi[x] for x in pad_texta])
        pad_textb = pad_sequnce([x for x in textb], DEFAULT_CONFIG['fix_length'])
        vec_textb = torch.tensor([self._word_vocab.stoi[x] for x in pad_textb])
        vec_predict = self._model(vec_texta.view(-1, 1), vec_textb.view(-1, 1))[0]
        soft_predict = torch.softmax(vec_predict, dim=0)
        predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=0)
        predict_class = self._label_vocab.itos[predict_index]
        predict_prob = predict_prob.item()
        return predict_prob, predict_class

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        shared_lstm = SharedLSTM(config)
        shared_lstm.load()
        self._model = shared_lstm
        self._word_vocab = config.word_vocab
        self._label_vocab = config.label_vocab

    def test(self, test_path):
        test_dataset = te_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        if not hasattr(LABEL, 'vocab'):
            LABEL.vocab = self._label_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = te_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        for dev_item in tqdm(dev_iter):
            item_score = te_tool.get_score(self._model, dev_item.texta, dev_item.textb, dev_item.label)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def deploy(self, route_path='/te', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/predict', methods=['POST', 'GET'])
        def predict():
            texta = request.args.get('texta', '')
            textb = request.args.get('textb', '')
            result = self.predict(texta, textb)
            return flask.jsonify({'state': 'OK', 'result': {'prob': result[0], 'class': result[1]}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class TextCNN(BaseModel):

    def __init__(self, args):
        super(TextCNN, self).__init__(args)
        self.class_num = args.class_num
        self.chanel_num = 1
        self.filter_num = args.filter_num
        self.filter_sizes = args.filter_sizes
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.embedding_dimension).from_pretrained(args.vectors)
            self.chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(self.chanel_num, self.filter_num, (size, self.embedding_dimension)) for size in self.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack((self.embedding(x), self.embedding2(x)), dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, int(item.size(2))).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class LSTMClassifier(BaseModel):

    def __init__(self, args):
        super(LSTMClassifier, self).__init__(args)
        self.hidden_dim = 300
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.embedding_dimension).from_pretrained(args.vectors)
        else:
            self.embedding2 = None
        self.lstm = nn.LSTM(self.embedding_dimension, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, self.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out
        final = lstm_out[-1]
        y = self.hidden2label(final)
        return y


def handle_line(entity1, entity2, sentence, begin_e1_token='<e1>', end_e1_token='</e1>', begin_e2_token='<e2>', end_e2_token='</e2>'):
    assert entity1 in sentence
    assert entity2 in sentence
    sentence = sentence.replace(entity1, begin_e1_token + entity1 + end_e1_token)
    sentence = sentence.replace(entity2, begin_e2_token + entity2 + end_e2_token)
    sentence = ' '.join(jieba.cut(sentence))
    sentence = sentence.replace('< e1 >', begin_e1_token)
    sentence = sentence.replace('< / e1 >', end_e1_token)
    sentence = sentence.replace('< e2 >', begin_e2_token)
    sentence = sentence.replace('< / e2 >', end_e2_token)
    return sentence.split()


class REDataset(Dataset):
    """Defines a Dataset of relation extraction format.
    eg:
    钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
    元武	元华	unknown	于师傅在一次京剧表演中，选了元龙（洪金宝）、元楼（元奎）、元彪、成龙、元华、元武、元泰7人担任七小福的主角。
    """

    def __init__(self, path, fields, encoding='utf-8', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                chunks = line.split()
                entity_1, entity_2, relation, sentence = tuple(chunks)
                sentence_list = handle_line(entity_1, entity_2, sentence)
                examples.append(Example.fromlist((sentence_list, relation), fields))
        super(REDataset, self).__init__(examples, fields, **kwargs)


class Attention(nn.Module):
    """
    several score types like dot,general and concat
    """

    def __init__(self, method='dot', hidden_size=None):
        super(Attention, self).__init__()
        self.method = method
        if self.method != 'dot':
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.W = nn.Linear(hidden_size, hidden_size)
            elif self.method == 'concat':
                self.W = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.rand(1, hidden_size))
                nn.init.xavier_normal_(self.v.data)

    def forward(self, query, key, value, mask=None, dropout=0):
        if self.method == 'general':
            scores = self.general(query, key)
        elif self.method == 'concat':
            scores = self.concat(query, key)
        else:
            scores = self.dot(query, key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if not dropout:
            p_attn = F.dropout(p_attn, dropout)
        return torch.matmul(p_attn, value), p_attn

    def dot(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores

    def general(self, query, key):
        scores = torch.matmul(self.W(query), key.transpose(-2, -1))
        return scores

    def concat(self, query, key):
        scores = torch.cat((query.expand(-1, key.size(1), -1), key), dim=2)
        scores = self.W(scores)
        scores = F.tanh(scores)
        scores = torch.matmul(scores, self.v.t()).transpose(-2, -1)
        return scores


class Decoder(nn.Module):

    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2, method='dot'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(method, hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word, last_hidden, encoder_outputs):
        embedded = self.embed(word).unsqueeze(1)
        embedded = self.dropout(embedded)
        context, attn_weights = self.attention(last_hidden[-1].unsqueeze(1), encoder_outputs, encoder_outputs)
        context = F.relu(context)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        output = torch.cat((output, context), 1)
        output = self.out(output)
        return output, hidden, attn_weights


class Encoder(nn.Module):
    """
    basic GRU encoder
    """

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, sentences, lengths, hidden=None):
        embedded = self.embed(sentences)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = trg.data[:, (0)]
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if teacher_force:
                decoder_input = trg.data[:, (t)].clone().detach()
            else:
                decoder_input = top1
        return outputs

    def predict(self, src, src_lens, sos, max_len):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1
        return outputs


class CBConfig(BaseConfig):

    def __init__(self, word_vocab, vector_path, **kwargs):
        super(CBConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class CBSeq2Seq(BaseModel):

    def __init__(self, args):
        super(CBSeq2Seq, self).__init__(args)
        self.args = args
        self.hidden_dim = args.embedding_dim
        self.vocabulary_size = args.vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        encoder = Encoder(vocabulary_size, embedding_dimension, self.hidden_dim, self.num_layers, self.dropout)
        decoder = Decoder(self.hidden_dim, embedding_dimension, vocabulary_size, self.num_layers, self.dropout, args.method)
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


class RNNLM(BaseModel):

    def __init__(self, args):
        super(RNNLM, self).__init__(args)
        self.args = args
        self.hidden_dim = args.embedding_dim
        self.vocabulary_size = args.vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout)
        self.bath_norm = nn.BatchNorm1d(embedding_dimension)
        self.hidden2label = nn.Linear(self.hidden_dim, self.vocabulary_size)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, sentence):
        x = self.embedding(sentence.to(DEVICE))
        self.hidden = self.init_hidden(batch_size=sentence.size(1))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.view(-1, lstm_out.size(2))
        lstm_out = self.bath_norm(lstm_out)
        y = self.hidden2label(lstm_out)
        return y


class LMConfig(BaseConfig):

    def __init__(self, word_vocab, vector_path, **kwargs):
        super(LMConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class LM(Module):

    def __init__(self):
        self._model = None
        self._word_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, log_dir=None, **kwargs):
        writer = SummaryWriter(log_dir=log_dir)
        train_dataset = lm_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = lm_tool.get_dataset(dev_path)
            word_vocab = lm_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab = lm_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        train_iter = lm_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'], bptt_len=DEFAULT_CONFIG['bptt_len'])
        config = LMConfig(word_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        rnnlm = RNNLM(config)
        self._model = rnnlm
        optim = torch.optim.Adam(rnnlm.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            rnnlm.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                optim.zero_grad()
                logits = rnnlm(item.text)
                item_loss = F.cross_entropy(logits, item.target.view(-1))
                acc_loss += item_loss.item()
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            writer.add_scalar('lm_train/acc_loss', acc_loss, epoch)
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
                writer.add_scalar('lm_train/dev_score', dev_score, epoch)
            writer.flush()
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        writer.close()
        config.save()
        rnnlm.save()

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = LMConfig.load(save_path)
        rnnlm = RNNLM(config)
        rnnlm.load()
        self._model = rnnlm
        self._word_vocab = config.word_vocab

    def test(self, test_path):
        test_dataset = lm_tool.get_dataset(test_path)
        if not hasattr(TEXT, 'vocab'):
            TEXT.vocab = self._word_vocab
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        dev_iter = lm_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'], bptt_len=DEFAULT_CONFIG['bptt_len'])
        for dev_item in tqdm(dev_iter):
            item_score = lm_tool.get_score(self._model, dev_item.text, dev_item.target)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)

    def _predict_next_word_max(self, sentence_list: list):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        pred_prob, pred_index = torch.max(torch.softmax(self._model(test_item)[-1], dim=0).cpu().data, dim=0)
        pred_word = TEXT.vocab.itos[pred_index]
        pred_prob = pred_prob.item()
        return pred_word, pred_prob

    def _predict_next_word_sample(self, sentence_list: list):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        pred_index = torch.multinomial(torch.softmax(self._model(test_item)[-1], dim=0).cpu().data, 1)
        pred_word = self._word_vocab.itos[pred_index]
        return pred_word

    def _predict_next_word_topk(self, sentence_list: list, topK=5):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        predict_softmax = torch.softmax(self._model(test_item)[-1], dim=0).cpu().data
        topK_prob, topK_index = torch.topk(predict_softmax, topK)
        topK_prob = topK_prob.tolist()
        topK_vocab = [self._word_vocab.itos[x] for x in topK_index]
        return list(zip(topK_vocab, topK_prob))

    def _predict_next_word_prob(self, sentence_list: list, next_word: str):
        test_item = torch.tensor([[self._word_vocab.stoi[x]] for x in sentence_list], device=DEVICE)
        predict_prob = torch.softmax(self._model(test_item)[-1], dim=0).cpu().data
        next_word_index = self._word_vocab.stoi[next_word]
        return predict_prob[next_word_index]

    def next_word(self, sentence: str, next_word: str):
        self._model.eval()
        temp_str = [x for x in light_tokenize(sentence)]
        predict_prob = self._predict_next_word_prob(temp_str, next_word)
        return predict_prob.item()

    def _next_word_score(self, sentence: str, next_word: str):
        self._model.eval()
        temp_str = [x for x in light_tokenize(sentence)]
        predict_prob = self._predict_next_word_prob(temp_str, next_word)
        return torch.log10(predict_prob).item()

    def next_word_topk(self, sentence: str, topK=5):
        self._model.eval()
        return self._predict_next_word_topk(sentence, topK)

    def sentence_score(self, sentence: str):
        self._model.eval()
        total_score = 0
        assert len(sentence) > 1
        for i in range(1, len(sentence)):
            temp_score = self._next_word_score(sentence[:i], sentence[i])
            total_score += temp_score
        return total_score

    def _predict_sentence(self, sentence: str, gen_len=30):
        results = []
        temp_str = [x for x in light_tokenize(sentence)]
        for i in range(gen_len):
            temp_result = self._predict_next_word_sample(temp_str)
            results.append(temp_result)
            temp_str.append(temp_result)
        return results

    def generate_sentence(self, sentence: str, gen_len=30):
        self._model.eval()
        results = self._predict_sentence(sentence, gen_len)
        predict_sen = ''.join([x for x in results])
        return sentence + predict_sen

    def deploy(self, route_path='/lm', host='localhost', port=None, debug=False):
        app = Flask(__name__)

        @app.route(route_path + '/next_word', methods=['POST', 'GET'])
        def next_word():
            sentence = request.args.get('sentence', '')
            word = request.args.get('word', '')
            result = self.next_word(sentence, word)
            return flask.jsonify({'state': 'OK', 'result': {'prob': result}})

        @app.route(route_path + '/generate_sentence', methods=['POST', 'GET'])
        def generate_sentence():
            sentence = request.args.get('sentence', '')
            gen_len = int(request.args.get('gen_len', 30))
            result = self.generate_sentence(sentence, gen_len)
            return flask.jsonify({'state': 'OK', 'result': {'sentence': result}})

        @app.route(route_path + '/next_word_topk', methods=['POST', 'GET'])
        def next_word_topk():
            sentence = request.args.get('sentence', '')
            topk = int(request.args.get('topk', 5))
            result = self.next_word_topk(sentence, topK=topk)
            return flask.jsonify({'state': 'OK', 'result': {'words': result}})

        @app.route(route_path + '/sentence_score', methods=['POST', 'GET'])
        def sentence_score():
            sentence = request.args.get('sentence', '')
            result = self.sentence_score(sentence)
            return flask.jsonify({'state': 'OK', 'result': {'score': result}})
        if not port:
            port = get_free_tcp_port()
        app.run(host=host, port=port, debug=debug)


class MTConfig(BaseConfig):

    def __init__(self, source_word_vocab, target_word_vocab, source_vector_path, target_vector_path, **kwargs):
        super(MTConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.source_word_vocab = source_word_vocab
        self.source_vocabulary_size = len(self.source_word_vocab)
        self.source_vector_path = source_vector_path
        self.target_word_vocab = target_word_vocab
        self.target_vocabulary_size = len(self.target_word_vocab)
        self.target_vector_path = target_vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class MTSeq2Seq(BaseModel):

    def __init__(self, args):
        super(MTSeq2Seq, self).__init__(args)
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.source_embedding_dim = args.source_embedding_dim
        self.target_embedding_dim = args.target_embedding_dim
        self.source_vector_path = args.source_vector_path
        self.target_vector_path = args.target_vector_path
        self.source_vocabulary_size = args.source_vocabulary_size
        self.target_vocabulary_size = args.target_vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        encoder = Encoder(self.source_vocabulary_size, self.source_embedding_dim, self.hidden_dim, self.num_layers, self.dropout)
        decoder = Decoder(self.hidden_dim, self.target_embedding_dim, self.target_vocabulary_size, self.num_layers, self.dropout, args.method)
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


def eng_tokenize(text):
    return nltk.word_tokenize(text)


SOURCE = Field(lower=True, tokenize=eng_tokenize, include_lengths=True, batch_first=True, init_token='<sos>', eos_token='<eos>')


TARGET = Field(lower=True, tokenize=light_tokenize, include_lengths=True, batch_first=True, init_token='<sos>', eos_token='<eos>')


class TSConfig(BaseConfig):

    def __init__(self, word_vocab, vector_path, **kwargs):
        super(TSConfig, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.vocabulary_size = len(self.word_vocab)
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)


class TSSeq2Seq(BaseModel):

    def __init__(self, args):
        super(TSSeq2Seq, self).__init__(args)
        self.args = args
        self.hidden_dim = args.embedding_dim
        self.vocabulary_size = args.vocabulary_size
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        encoder = Encoder(vocabulary_size, embedding_dimension, self.hidden_dim, self.num_layers, self.dropout)
        decoder = Decoder(self.hidden_dim, embedding_dimension, vocabulary_size, self.num_layers, self.dropout, args.method)
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


class CBOWBase(BaseModel):

    def __init__(self, args):
        super(CBOWBase, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size)

    def forward(self, context):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return target_embedding

    def loss(self, context, target):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return F.cross_entropy(target_embedding, target.view(-1))


def default_tokenize(sentence):
    return list(jieba.cut(sentence))


class CBOWDataset(Dataset):

    def __init__(self, path, fields, window_size=3, tokenize=default_tokenize, encoding='utf-8', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                words = tokenize(line.strip())
                if len(words) < window_size + 1:
                    continue
                for i in range(len(words)):
                    example = words[max(0, i - window_size):i] + words[min(i + 1, len(words)):min(len(words), i + window_size) + 1], words[i]
                    examples.append(Example.fromlist(example, fields))
        super(CBOWDataset, self).__init__(examples, fields, **kwargs)


class CBOWHierarchicalSoftmax(BaseModel):

    def __init__(self, args):
        super(CBOWHierarchicalSoftmax, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(2 * self.vocabulary_size - 1, self.embedding_dimension, sparse=True)
        self.context_embeddings = nn.Embedding(2 * self.vocabulary_size - 1, self.embedding_dimension, sparse=True)

    def forward(self, x):
        pass

    def loss(self, pos_context, pos_path, neg_context, neg_path):
        pass


class HuffmanNode:

    def __init__(self, word_id, frequency):
        self.word_id = word_id
        self.frequency = frequency
        self.left_child = None
        self.right_child = None
        self.father = None
        self.Huffman_code = []
        self.path = []


class HuffmanTree:

    def __init__(self, wordid_frequency_dict):
        self.word_count = len(wordid_frequency_dict)
        self.wordid_code = dict()
        self.wordid_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(wordid, frequency) for wordid, frequency in wordid_frequency_dict.items()]
        self.huffman = [HuffmanNode(wordid, frequency) for wordid, frequency in wordid_frequency_dict.items()]
        self.build_tree(unmerge_node_list)
        self.generate_huffman_code_and_path()

    def merge_node(self, node1, node2):
        sum_frequency = node1.frequency + node2.frequency
        mid_node_id = len(self.huffman)
        father_node = HuffmanNode(mid_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        while len(node_list) > 1:
            i1 = 0
            i2 = 1
            if node_list[i2].frequency < node_list[i1].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def generate_huffman_code_and_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left_child or node.right_child:
                code = node.Huffman_code
                path = node.path
                node.left_child.Huffman_code = code + [1]
                node.right_child.Huffman_code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]
                stack.append(node.right_child)
                node = node.left_child
            word_id = node.word_id
            word_code = node.Huffman_code
            word_path = node.path
            self.huffman[word_id].Huffman_code = word_code
            self.huffman[word_id].path = word_path
            self.wordid_code[word_id] = word_code
            self.wordid_path[word_id] = word_path

    def get_all_pos_and_neg_path(self):
        positive = []
        negative = []
        for word_id in range(self.word_count):
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.huffman[word_id].Huffman_code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative


class CBOWNegativeSampling(BaseModel):

    def __init__(self, args):
        super(CBOWNegativeSampling, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.context_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)

    def forward(self, context, target):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.context_embeddings(target)
        target_score = torch.bmm(target_embedding, context_embedding.unsqueeze(2))
        return torch.sigmoid(target_score)

    def loss(self, context, pos, neg):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        pos_embedding = self.context_embeddings(pos)
        neg_embedding = self.context_embeddings(neg).squeeze()
        pos_score = torch.bmm(pos_embedding, context_embedding.unsqueeze(2)).squeeze()
        neg_score = torch.bmm(neg_embedding, context_embedding.unsqueeze(2)).squeeze()
        pos_score = torch.sum(F.logsigmoid(pos_score), dim=0)
        neg_score = torch.sum(F.logsigmoid(-1 * neg_score), dim=0)
        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))


class Sampling(object):

    def __init__(self, vocab: Vocab, weight=0.75):
        self.vocab = vocab
        self.weight = weight
        self.weighted_list = [(self.vocab.freqs[s] ** self.weight) for s in self.vocab.itos]

    def sampling(self, num):
        return torch.multinomial(torch.tensor(self.weighted_list), num).tolist()


class SkipGramBase(BaseModel):

    def __init__(self, args):
        super(SkipGramBase, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size)

    def forward(self, target):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).squeeze()
        return context_embedding

    def loss(self, target, context):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).reshape(target_embedding.size(0), -1)
        return F.cross_entropy(context_embedding, context.view(-1))


class SkipGramDataset(Dataset):

    def __init__(self, path, fields, window_size=3, tokenize=default_tokenize, encoding='utf-8', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                words = tokenize(line.strip())
                if len(words) < window_size + 1:
                    continue
                for i in range(len(words)):
                    contexts = words[max(0, i - window_size):i] + words[min(i + 1, len(words)):min(len(words), i + window_size) + 1]
                    for context in contexts:
                        examples.append(Example.fromlist((context, words[i]), fields))
        super(SkipGramDataset, self).__init__(examples, fields, **kwargs)


class SkipGramHierarchicalSoftmax(BaseModel):

    def __init__(self, args):
        super(SkipGramHierarchicalSoftmax, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(2 * self.vocabulary_size - 1, self.embedding_dimension, sparse=True)
        self.context_embeddings = nn.Embedding(2 * self.vocabulary_size - 1, self.embedding_dimension, sparse=True)

    def forward(self, pos_target, pos_path, neg_target, neg_path):
        pos_target_embedding = torch.sum(self.word_embeddings(pos_target), dim=1, keepdim=True)
        pos_path_embedding = self.context_embeddings(pos_path)
        pos_score = torch.bmm(pos_target_embedding, pos_path_embedding.transpose(2, 1)).squeeze()
        neg_target_embedding = torch.sum(self.word_embeddings(neg_target), dim=1, keepdim=True)
        neg_path_embedding = self.context_embeddings(neg_path)
        neg_score = torch.bmm(neg_target_embedding, neg_path_embedding.transpose(2, 1)).squeeze()
        pos_sigmoid_score = torch.lt(torch.sigmoid(pos_score), 0.5)
        neg_sigmoid_score = torch.gt(torch.sigmoid(neg_score), 0.5)
        sigmoid_score = torch.cat((pos_sigmoid_score, neg_sigmoid_score))
        sigmoid_score = torch.sum(sigmoid_score, dim=0).item() / sigmoid_score.size(0)
        return sigmoid_score

    def loss(self, pos_target, pos_path, neg_target, neg_path):
        pos_target_embedding = torch.sum(self.word_embeddings(pos_target), dim=1, keepdim=True)
        pos_path_embedding = self.context_embeddings(pos_path)
        pos_score = torch.bmm(pos_target_embedding, pos_path_embedding.transpose(2, 1)).squeeze()
        neg_target_embedding = torch.sum(self.word_embeddings(neg_target), dim=1, keepdim=True)
        neg_path_embedding = self.context_embeddings(neg_path)
        neg_score = torch.bmm(neg_target_embedding, neg_path_embedding.transpose(2, 1)).squeeze()
        pos_score = torch.sum(F.logsigmoid(-1 * pos_score))
        neg_score = torch.sum(F.logsigmoid(neg_score))
        loss = -1 * (pos_score + neg_score)
        return loss


class SkipGramNegativeSampling(BaseModel):

    def __init__(self, args):
        super(SkipGramNegativeSampling, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.context_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dimension)

    def forward(self, target, context):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.context_embeddings(context)
        target_score = torch.matmul(target_embedding, context_embedding.transpose(2, 1)).squeeze()
        return torch.sigmoid(target_score)

    def loss(self, target, pos, neg):
        target_embedding = self.word_embeddings(target)
        pos_embedding = self.context_embeddings(pos)
        neg_embedding = self.context_embeddings(neg).squeeze()
        pos_score = torch.matmul(target_embedding, pos_embedding.transpose(2, 1)).squeeze()
        neg_score = torch.matmul(target_embedding, neg_embedding.transpose(1, 2)).squeeze()
        pos_score = torch.sum(F.logsigmoid(pos_score), dim=0)
        neg_score = torch.sum(F.logsigmoid(-1 * neg_score), dim=0)
        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Biaffine,
     lambda: ([], {'n_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CBOWBase,
     lambda: ([], {'args': _mock_config(save_path=4, vocabulary_size=4, embedding_dim=4)}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     True),
    (CBOWHierarchicalSoftmax,
     lambda: ([], {'args': _mock_config(save_path=4, vocabulary_size=4, embedding_dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBOWNegativeSampling,
     lambda: ([], {'args': _mock_config(save_path=4, vocabulary_size=4, embedding_dim=4)}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (Decoder,
     lambda: ([], {'embed_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.rand([1, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'input_size': 4, 'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (IndependentDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_hidden': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SharedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SkipGramBase,
     lambda: ([], {'args': _mock_config(save_path=4, vocabulary_size=4, embedding_dim=4)}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     True),
    (SkipGramNegativeSampling,
     lambda: ([], {'args': _mock_config(save_path=4, vocabulary_size=4, embedding_dim=4)}),
     lambda: ([torch.ones([4, 4, 4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (VanillaWordEmbeddingLookup,
     lambda: ([], {'vocabulary_size': 4, 'embedding_dim': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     False),
]

class Test_smilelight_lightNLP(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

