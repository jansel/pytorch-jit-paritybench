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
utils = _module
convert = _module
ner = _module
model = _module
pos = _module
model = _module
srl = _module
model = _module
sp = _module
gdp = _module
components = _module
biaffine = _module
dropout = _module
lstm = _module
mlp = _module
model = _module
module = _module
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
feature_extractor = _module
parser_state = _module
vectors = _module
sr = _module
ss = _module
model = _module
module = _module
pad = _module
te = _module
model = _module
module = _module
tc = _module
re = _module
model = _module
module = _module
preprocess = _module
sa = _module
model = _module
module = _module
tg = _module
cb = _module
models = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
lm = _module
model = _module
module = _module
mt = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
ts = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
module = _module
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
model = _module
negative_sampling = _module
model = _module
huffman_tree = _module
sampling = _module
skip_gram = _module
model = _module
module = _module
model = _module
model = _module
model = _module
module = _module
setup = _module
test_flask = _module

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


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import PackedSequence


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


from collections import Counter


from collections import deque


import random


from torch.nn.utils import clip_grad_norm_


from typing import List


LEVEL_COLOR = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
    'ERROR': 'red', 'CRITICAL': 'red,bg_white'}


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {'black': '30', 'red': '31', 'green': '32', 'yellow': '33',
        'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37',
        'bg_black': '40', 'bg_red': '41', 'bg_green': '42', 'bg_yellow':
        '43', 'bg_blue': '44', 'bg_magenta': '45', 'bg_cyan': '46',
        'bg_white': '47', 'light_black': '1;30', 'light_red': '1;31',
        'light_green': '1;32', 'light_yellow': '1;33', 'light_blue': '1;34',
        'light_magenta': '1;35', 'light_cyan': '1;36', 'light_white':
        '1;37', 'light_bg_black': '100', 'light_bg_red': '101',
        'light_bg_green': '102', 'light_bg_yellow': '103', 'light_bg_blue':
        '104', 'light_bg_magenta': '105', 'light_bg_cyan': '106',
        'light_bg_white': '107'}

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


FILE_LOG_FMT = (
    '[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    )


STDOUT_DATE_FMT = '%Y-%m-%d %H:%M:%S'


STDOUT_LOG_FMT = (
    '%(log_color)s[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    )


def _get_logger(log_to_file=True, log_filename='default.log', log_level='DEBUG'
    ):
    _logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(ColoredFormatter(fmt=STDOUT_LOG_FMT,
        datefmt=STDOUT_DATE_FMT))
    _logger.addHandler(stdout_handler)
    if log_to_file:
        _tmp_path = os.path.dirname(os.path.abspath(__file__))
        _tmp_path = os.path.join(_tmp_path, '../logs/{}'.format(log_filename))
        file_handler = logging.handlers.TimedRotatingFileHandler(_tmp_path,
            when='midnight', backupCount=30)
        file_formatter = logging.Formatter(fmt=FILE_LOG_FMT, datefmt=
            FILE_DATE_FMT)
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


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in +
            bias_y))
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

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
        bidirectional=False):
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
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                    hidden_size=hidden_size))
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
                mask = torch.cat([mask[:batch_size] for batch_size in
                    batch_sizes])
                x *= mask
            x = torch.split(x, batch_sizes.tolist())
            f_output = self.layer_forward(x=x, hx=hx, cell=self.f_cells[
                layer], batch_sizes=batch_sizes, reverse=False)
            if self.bidirectional:
                b_output = self.layer_forward(x=x, hx=hx, cell=self.b_cells
                    [layer], batch_sizes=batch_sizes, reverse=True)
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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.word_embedding = nn.Embedding(vocabulary_size, word_dim).to(DEVICE
            )
        vectors = Vectors(args.vector_path).vectors
        self.pretrained_embedding = nn.Embedding.from_pretrained(vectors).to(
            DEVICE)
        self.pos_embedding = nn.Embedding(pos_num, pos_dim).to(DEVICE)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout).to(DEVICE
            )
        self.lstm = LSTM(word_dim + pos_dim, self.hidden_dim, bidirectional
            =self.bidirectional, num_layers=self.lstm_layters, dropout=args
            .lstm_dropout).to(DEVICE)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout).to(DEVICE)
        self.mlp_arc_h = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.
            mlp_arc, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_arc_d = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.
            mlp_arc, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_rel_h = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.
            mlp_rel, dropout=args.mlp_dropout).to(DEVICE)
        self.mlp_rel_d = MLP(n_in=args.lstm_hidden * 2, n_hidden=args.
            mlp_rel, dropout=args.mlp_dropout).to(DEVICE)
        self.arc_attn = Biaffine(n_in=args.mlp_arc, bias_x=True, bias_y=False
            ).to(DEVICE)
        self.rel_attn = Biaffine(n_in=args.mlp_rel, n_out=args.ref_num,
            bias_x=True, bias_y=True).to(DEVICE)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embedding.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        return h0, c0

    def forward(self, words, tags):
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        embed = self.pretrained_embedding(words)
        embed += self.word_embedding(words.masked_fill_(words.ge(self.
            word_embedding.num_embeddings), 0))
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
            raise AttributeError(
                'config object must init save_path attr in init method!')
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
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dimension
            ).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path)
                )
            vectors = Vectors(args.vector_path).vectors
            self.word_embedding = nn.Embedding.from_pretrained(vectors,
                freeze=not args.non_static).to(DEVICE)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim).to(DEVICE)
        self.lstm = nn.LSTM(embedding_dimension + pos_dim + 1, self.
            hidden_dim // 2, bidirectional=self.bidirectional, num_layers=
            self.num_layers, dropout=self.dropout).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(DEVICE)
        self.crflayer = CRF(self.tag_num).to(DEVICE)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
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
        word = self.word_embedding(sentence.to(DEVICE)).to(DEVICE)
        pos = self.pos_embedding(poses.to(DEVICE)).to(DEVICE)
        rels = rels.view(rels.size(0), rels.size(1), 1).float().to(DEVICE)
        x = torch.cat((word, pos, rels), dim=2)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE))
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def bis_pos(words, tags):
    assert len(words) == len(tags)
    poses = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] in ['B', 'S']:
            begin = i
            temp_type = tag.split('-')[1]
        if i == len(tags) - 1:
            poses.append((''.join(words[begin:i + 1]), temp_type))
        elif tags[i + 1].split('-')[0] != 'I' or tags[i + 1].split('-')[1
            ] != temp_type:
            poses.append((''.join(words[begin:i + 1]), temp_type))
            begin = i + 1
            temp_type = tags[i + 1].split('-')[1]
    return poses


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def light_tokenize(sequence: str):
    return [sequence]


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
        self.linear1 = nn.Linear(input_dim, self.hidden_dim).to(DEVICE)
        self.linear2 = nn.Linear(self.hidden_dim, Actions.NUM_ACTIONS).to(
            DEVICE)

    def forward(self, inputs):
        input_vec = vectors.concat_and_flatten(inputs)
        temp_vec = self.linear1(input_vec)
        temp_vec = F.relu(temp_vec).to(DEVICE)
        result = self.linear2(temp_vec)
        return result


class MLPCombinerNetwork(nn.Module):

    def __init__(self, embedding_dim):
        super(MLPCombinerNetwork, self).__init__()
        self.linear1 = nn.Linear(embedding_dim * 2, embedding_dim).to(DEVICE)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim).to(DEVICE)

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
        self.linear = nn.Linear(self.embedding_dim * 2, self.embedding_dim).to(
            DEVICE)
        self.hidden_dim = self.embedding_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=
            self.num_layers, dropout=dropout).to(DEVICE)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            DEVICE)
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

    def __init__(self, vocabulary_size, embedding_dim, vector_path=None,
        non_static=False):
        super(VanillaWordEmbeddingLookup, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.
            embedding_dim).to(DEVICE)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(
                word_vectors, freeze=not non_static).to(DEVICE)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.to(DEVICE)).to(DEVICE)
        return embeds


class BiLSTMWordEmbeddingLookup(nn.Module):

    def __init__(self, vocabulary_size, word_embedding_dim, hidden_dim,
        num_layers, dropout, vector_path=None, non_static=False):
        super(BiLSTMWordEmbeddingLookup, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.
            word_embedding_dim).to(DEVICE)
        if vector_path:
            logger.info('logging word vectors from {}'.format(vector_path))
            word_vectors = Vectors(vector_path).vectors
            self.word_embeddings = self.word_embeddings.from_pretrained(
                word_vectors, freeze=not non_static).to(DEVICE)
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim // 2,
            bidirectional=True, num_layers=num_layers, dropout=dropout).to(
            DEVICE)
        self.hidden = self.init_hidden()

    def forward(self, sentence):
        embeddings = self.word_embeddings(sentence)
        lstm_hiddens, self.hidden = self.lstm(embeddings, self.hidden)
        return lstm_hiddens

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        return h0, c0

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()


def action_tokenize(sequence: str):
    return [sequence]


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
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(
            DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path)
                )
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze
                =not args.non_static).to(DEVICE)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2,
            bidirectional=self.bidirectional, num_layers=self.num_layers,
            dropout=self.dropout).to(DEVICE)
        self.dropout_layer = nn.Dropout(self.dropout).to(DEVICE)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim * 2).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.class_num).to(
            DEVICE)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2
            ).to(DEVICE)
        return h0, c0

    def forward(self, left, right):
        left_vec = self.embedding(left.to(DEVICE)).to(DEVICE)
        right_vec = self.embedding(right.to(DEVICE)).to(DEVICE)
        self.hidden = self.init_hidden(batch_size=left.size(1))
        left_lstm_out, (left_lstm_hidden, _) = self.lstm(left_vec, self.hidden)
        right_lstm_out, (right_lstm_hidden, _) = self.lstm(right_vec, self.
            hidden)
        merged = torch.cat((left_lstm_out[-1], right_lstm_out[-1]), dim=1)
        merged = self.dropout_layer(merged)
        merged = self.batch_norm(merged)
        predict = self.hidden2label(merged)
        return predict


class TextCNN(BaseModel):

    def __init__(self, args):
        super(TextCNN, self).__init__(args)
        self.class_num = args.class_num
        self.chanel_num = 1
        self.filter_num = args.filter_num
        self.filter_sizes = args.filter_sizes
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.
            embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path)
                )
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze
                =not args.non_static).to(DEVICE)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.
                embedding_dimension).from_pretrained(args.vectors).to(DEVICE)
            self.chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(self.chanel_num, self.
            filter_num, (size, self.embedding_dimension)) for size in self.
            filter_sizes]).to(DEVICE)
        self.dropout = nn.Dropout(args.dropout).to(DEVICE)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.
            class_num).to(DEVICE)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack((self.embedding(x), self.embedding2(x)), dim=1).to(
                DEVICE)
        else:
            x = self.embedding(x).to(DEVICE)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, int(item.size(2))).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def handle_line(entity1, entity2, sentence, begin_e1_token='<e1>',
    end_e1_token='</e1>', begin_e2_token='<e2>', end_e2_token='</e2>'):
    assert entity1 in sentence
    assert entity2 in sentence
    sentence = sentence.replace(entity1, begin_e1_token + entity1 +
        end_e1_token)
    sentence = sentence.replace(entity2, begin_e2_token + entity2 +
        end_e2_token)
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
                examples.append(Example.fromlist((sentence_list, relation),
                    fields))
        super(REDataset, self).__init__(examples, fields, **kwargs)


def pad_sequnce(sequence, seq_length, pad_token='<pad>'):
    padded_seq = sequence[:]
    if len(padded_seq) < seq_length:
        padded_seq.extend([pad_token for _ in range(len(padded_seq),
            seq_length)])
    return padded_seq[:seq_length]


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

    def __init__(self, embed_size, hidden_size, output_size, n_layers=1,
        dropout=0.2, method='dot'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(output_size, embed_size).to(DEVICE)
        self.dropout = nn.Dropout(dropout, inplace=True).to(DEVICE)
        self.attention = Attention(method, hidden_size).to(DEVICE)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers,
            dropout=dropout, batch_first=True).to(DEVICE)
        self.out = nn.Linear(hidden_size * 2, output_size).to(DEVICE)

    def forward(self, word, last_hidden, encoder_outputs):
        embedded = self.embed(word).unsqueeze(1)
        embedded = self.dropout(embedded)
        context, attn_weights = self.attention(last_hidden[-1].unsqueeze(1),
            encoder_outputs, encoder_outputs)
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

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1,
        dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size).to(DEVICE)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=
            True, dropout=dropout, bidirectional=True).to(DEVICE)

    def forward(self, sentences, lengths, hidden=None):
        embedded = self.embed(sentences)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.
            hidden_size:]
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
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = trg.data[:, (0)]
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if teacher_force:
                decoder_input = trg.data[:, (t)].clone().detach().to(DEVICE)
            else:
                decoder_input = top1.to(DEVICE)
        return outputs

    def predict(self, src, src_lens, sos, max_len):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1.to(DEVICE)
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
        encoder = Encoder(vocabulary_size, embedding_dimension, self.
            hidden_dim, self.num_layers, self.dropout).to(DEVICE)
        decoder = Decoder(self.hidden_dim, embedding_dimension,
            vocabulary_size, self.num_layers, self.dropout, args.method).to(
            DEVICE)
        self.seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


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
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(
            DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path)
                )
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze
                =not args.non_static).to(DEVICE)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim,
            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        self.bath_norm = nn.BatchNorm1d(embedding_dimension).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.vocabulary_size
            ).to(DEVICE)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            DEVICE)
        return h0, c0

    def forward(self, sentence):
        x = self.embedding(sentence.to(DEVICE)).to(DEVICE)
        self.hidden = self.init_hidden(batch_size=sentence.size(1))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.view(-1, lstm_out.size(2))
        lstm_out = self.bath_norm(lstm_out)
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)


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

    def __init__(self, embed_size, hidden_size, output_size, n_layers=1,
        dropout=0.2, method='dot'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(output_size, embed_size).to(DEVICE)
        self.dropout = nn.Dropout(dropout, inplace=True).to(DEVICE)
        self.attention = Attention(method, hidden_size).to(DEVICE)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers,
            dropout=dropout, batch_first=True).to(DEVICE)
        self.out = nn.Linear(hidden_size * 2, output_size).to(DEVICE)

    def forward(self, word, last_hidden, encoder_outputs):
        embedded = self.embed(word).unsqueeze(1)
        embedded = self.dropout(embedded)
        context, attn_weights = self.attention(last_hidden[-1].unsqueeze(1),
            encoder_outputs, encoder_outputs)
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

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1,
        dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size).to(DEVICE)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=
            True, dropout=dropout, bidirectional=True).to(DEVICE)

    def forward(self, sentences, lengths, hidden=None):
        embedded = self.embed(sentences)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.
            hidden_size:]
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
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = trg.data[:, (0)]
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if teacher_force:
                decoder_input = trg.data[:, (t)].clone().detach().to(DEVICE)
            else:
                decoder_input = top1.to(DEVICE)
        return outputs

    def predict(self, src, src_lens, sos, max_len):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1.to(DEVICE)
        return outputs


class MTConfig(BaseConfig):

    def __init__(self, source_word_vocab, target_word_vocab,
        source_vector_path, target_vector_path, **kwargs):
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
        encoder = Encoder(self.source_vocabulary_size, self.
            source_embedding_dim, self.hidden_dim, self.num_layers, self.
            dropout).to(DEVICE)
        decoder = Decoder(self.hidden_dim, self.target_embedding_dim, self.
            target_vocabulary_size, self.num_layers, self.dropout, args.method
            ).to(DEVICE)
        self.seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


def eng_tokenize(text):
    return nltk.word_tokenize(text)


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

    def __init__(self, embed_size, hidden_size, output_size, n_layers=1,
        dropout=0.2, method='dot'):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(output_size, embed_size).to(DEVICE)
        self.dropout = nn.Dropout(dropout, inplace=True).to(DEVICE)
        self.attention = Attention(method, hidden_size).to(DEVICE)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers,
            dropout=dropout, batch_first=True).to(DEVICE)
        self.out = nn.Linear(hidden_size * 2, output_size).to(DEVICE)

    def forward(self, word, last_hidden, encoder_outputs):
        embedded = self.embed(word).unsqueeze(1)
        embedded = self.dropout(embedded)
        context, attn_weights = self.attention(last_hidden[-1].unsqueeze(1),
            encoder_outputs, encoder_outputs)
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

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1,
        dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size).to(DEVICE)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=
            True, dropout=dropout, bidirectional=True).to(DEVICE)

    def forward(self, sentences, lengths, hidden=None):
        embedded = self.embed(sentences)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.
            hidden_size:]
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
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = trg.data[:, (0)]
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if teacher_force:
                decoder_input = trg.data[:, (t)].clone().detach().to(DEVICE)
            else:
                decoder_input = top1.to(DEVICE)
        return outputs

    def predict(self, src, src_lens, sos, max_len):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)
        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input,
                hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1.to(DEVICE)
        return outputs


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
        encoder = Encoder(vocabulary_size, embedding_dimension, self.
            hidden_dim, self.num_layers, self.dropout).to(DEVICE)
        decoder = Decoder(self.hidden_dim, embedding_dimension,
            vocabulary_size, self.num_layers, self.dropout, args.method).to(
            DEVICE)
        self.seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return self.seq2seq(src, trg, teacher_forcing_ratio)

    def predict(self, src, src_lens, sos, max_len):
        return self.seq2seq.predict(src, src_lens, sos, max_len)


class CBOWBase(BaseModel):

    def __init__(self, args):
        super(CBOWBase, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.
            embedding_dimension).to(DEVICE)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size
            ).to(DEVICE)

    def forward(self, context):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return target_embedding

    def loss(self, context, target):
        context_embedding = torch.sum(self.word_embeddings(context), dim=1)
        target_embedding = self.linear(context_embedding)
        return F.cross_entropy(target_embedding, target.view(-1))


ROOT = '<ROOT>'


class SkipGramBase(BaseModel):

    def __init__(self, args):
        super(SkipGramBase, self).__init__(args)
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.
            embedding_dimension).to(DEVICE)
        self.linear = nn.Linear(self.embedding_dimension, self.vocabulary_size
            ).to(DEVICE)

    def forward(self, target):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).squeeze()
        return context_embedding

    def loss(self, target, context):
        target_embedding = self.word_embeddings(target)
        context_embedding = self.linear(target_embedding).reshape(
            target_embedding.size(0), -1)
        return F.cross_entropy(context_embedding, context.view(-1))


def default_tokenize(sentence):
    return list(jieba.cut(sentence))


class SkipGramDataset(Dataset):

    def __init__(self, path, fields, window_size=3, tokenize=
        default_tokenize, encoding='utf-8', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                words = tokenize(line.strip())
                if len(words) < window_size + 1:
                    continue
                for i in range(len(words)):
                    contexts = words[max(0, i - window_size):i] + words[min
                        (i + 1, len(words)):min(len(words), i + window_size
                        ) + 1]
                    for context in contexts:
                        examples.append(Example.fromlist((context, words[i]
                            ), fields))
        super(SkipGramDataset, self).__init__(examples, fields, **kwargs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_smilelight_lightNLP(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Attention(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Biaffine(*[], **{'n_in': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(IndependentDropout(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MLP(*[], **{'n_in': 4, 'n_hidden': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(SharedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

