import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
supar = _module
cmds = _module
aj_con = _module
biaffine_dep = _module
biaffine_sdp = _module
cmd = _module
crf2o_dep = _module
crf_con = _module
crf_dep = _module
vi_con = _module
vi_dep = _module
vi_sdp = _module
model = _module
models = _module
const = _module
aj = _module
model = _module
parser = _module
crf = _module
model = _module
parser = _module
vi = _module
model = _module
parser = _module
dep = _module
biaffine = _module
model = _module
parser = _module
model = _module
parser = _module
crf2o = _module
model = _module
parser = _module
model = _module
parser = _module
sdp = _module
model = _module
parser = _module
model = _module
parser = _module
modules = _module
affine = _module
dropout = _module
gnn = _module
lstm = _module
mlp = _module
pretrained = _module
transformer = _module
parser = _module
structs = _module
chain = _module
dist = _module
fn = _module
semiring = _module
tree = _module
vi = _module
utils = _module
common = _module
config = _module
data = _module
embed = _module
field = _module
fn = _module
logging = _module
metric = _module
optim = _module
parallel = _module
tokenizer = _module
transform = _module
vocab = _module
test_fn = _module
test_parse = _module
test_struct = _module
test_transform = _module

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


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from typing import List


from typing import Tuple


from typing import Dict


from typing import Iterable


from typing import Set


from typing import Union


from typing import Callable


from typing import Optional


from torch.nn.modules.rnn import apply_permutation


from torch.nn.utils.rnn import PackedSequence


import copy


import torch.nn.functional as F


from typing import Any


from torch.cuda.amp import GradScaler


from torch.optim import Adam


from torch.optim.lr_scheduler import ExponentialLR


from torch.distributions.utils import lazy_property


import torch.autograd as autograd


from torch.distributions.distribution import Distribution


from torch.autograd import Function


import itertools


from functools import reduce


import queue


from collections import Counter


from collections import defaultdict


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import functools


import re


from typing import TYPE_CHECKING


def extract(path: str, reload: bool=False, clean: bool=False) ->str:
    extracted = path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.infolist()[0].filename)
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.getnames()[0])
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif path.endswith('.gz'):
        extracted = path[:-3]
        with gzip.open(path) as fgz:
            with open(extracted, 'wb') as f:
                shutil.copyfileobj(fgz, f)
    if clean:
        os.remove(path)
    return extracted


def gather(obj: Any) ->Iterable[Any]:
    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)
    return objs


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return not is_dist() or dist.get_rank() == 0


def wait(fn) ->Any:

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        value = None
        if is_master():
            value = fn(*args, **kwargs)
        if is_dist():
            dist.barrier()
            value = gather(value)[0]
        return value
    return wrapper


@wait
def download(url: str, path: Optional[str]=None, reload: bool=False, clean: bool=False) ->str:
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    if path is None:
        path = CACHE
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    if reload and os.path.exists(path):
        os.remove(path)
    if not os.path.exists(path):
        sys.stderr.write(f'Downloading {url} to {path}\n')
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except (ValueError, urllib.error.URLError):
            raise RuntimeError(f'File {url} unavailable. Please try other sources.')
    return extract(path, reload, clean)


class Config(object):

    def __init__(self, **kwargs):
        super(Config, self).__init__()
        self.update(kwargs)

    def __repr__(self):
        return OmegaConf.to_yaml(vars(self))

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def update(self, kwargs):
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    def pop(self, key, val=None):
        return self.__dict__.pop(key, val)

    @classmethod
    def load(cls, conf='', unknown=None, **kwargs):
        config = ConfigParser()
        config.read(conf if not conf or os.path.exists(conf) else download(supar.CONFIG['github'].get(conf, conf)))
        config = dict((name, literal_eval(value)) for section in config.sections() for name, value in config.items(section))
        if unknown is not None:
            parser = argparse.ArgumentParser()
            for name, value in config.items():
                parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
            config.update(vars(parser.parse_args(unknown)))
        config.update(kwargs)
        return cls(**config)


def pad(tensors: List[torch.Tensor], padding_value: int=0, total_length: int=None, padding_side: str='right') ->torch.Tensor:
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors) for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[(slice(-i, None) if padding_side == 'left' else slice(0, i)) for i in tensor.size()]] = tensor
    return out_tensor


class SinusoidPositionalEmbedding(nn.Module):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[:, 0::2], pos[:, 1::2] = pos[:, 0::2].sin(), pos[:, 1::2].cos()
        return pos


class SinusoidRelativePositionalEmbedding(nn.Module):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len))
        pos = (pos - pos.unsqueeze(-1)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)).div(2, rounding_mode='floor') * 2 / n_model)
        pos[..., 0::2], pos[..., 1::2] = pos[..., 0::2].sin(), pos[..., 1::2].cos()
        return pos


class Model(nn.Module):

    def __init__(self, n_words, n_tags=None, n_chars=None, n_lemmas=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, char_dropout=0, elmo_bos_eos=(True, True), elmo_dropout=0.5, bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, encoder_dropout=0.33, pad_index=0, **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        if encoder == 'lstm':
            self.word_embed = nn.Embedding(num_embeddings=self.args.n_words, embedding_dim=self.args.n_embed)
            n_input = self.args.n_embed
            if self.args.n_pretrained != self.args.n_embed:
                n_input += self.args.n_pretrained
            if 'tag' in self.args.feat:
                self.tag_embed = nn.Embedding(num_embeddings=self.args.n_tags, embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'char' in self.args.feat:
                self.char_embed = CharLSTM(n_chars=self.args.n_chars, n_embed=self.args.n_char_embed, n_hidden=self.args.n_char_hidden, n_out=self.args.n_feat_embed, pad_index=self.args.char_pad_index, dropout=self.args.char_dropout)
                n_input += self.args.n_feat_embed
            if 'lemma' in self.args.feat:
                self.lemma_embed = nn.Embedding(num_embeddings=self.args.n_lemmas, embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'elmo' in self.args.feat:
                self.elmo_embed = ELMoEmbedding(n_out=self.args.n_plm_embed, bos_eos=self.args.elmo_bos_eos, dropout=self.args.elmo_dropout, finetune=self.args.finetune)
                n_input += self.elmo_embed.n_out
            if 'bert' in self.args.feat:
                self.bert_embed = TransformerEmbedding(name=self.args.bert, n_layers=self.args.n_bert_layers, n_out=self.args.n_plm_embed, pooling=self.args.bert_pooling, pad_index=self.args.bert_pad_index, mix_dropout=self.args.mix_dropout, finetune=self.args.finetune)
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=self.args.embed_dropout)
            self.encoder = VariationalLSTM(input_size=n_input, hidden_size=self.args.n_encoder_hidden // 2, num_layers=self.args.n_encoder_layers, bidirectional=True, dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
        elif encoder == 'transformer':
            self.word_embed = TransformerWordEmbedding(n_vocab=self.args.n_words, n_embed=self.args.n_embed, pos=self.args.pos, pad_index=self.args.pad_index)
            self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
            self.encoder = TransformerEncoder(layer=TransformerEncoderLayer(n_heads=self.args.n_encoder_heads, n_model=self.args.n_encoder_hidden, n_inner=self.args.n_encoder_inner, attn_dropout=self.args.encoder_attn_dropout, ffn_dropout=self.args.encoder_ffn_dropout, dropout=self.args.encoder_dropout), n_layers=self.args.n_encoder_layers, n_model=self.args.n_encoder_hidden)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
        elif encoder == 'bert':
            self.encoder = TransformerEmbedding(name=self.args.bert, n_layers=self.args.n_bert_layers, pooling=self.args.bert_pooling, pad_index=self.args.pad_index, mix_dropout=self.args.mix_dropout, finetune=True)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_encoder_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats=None):
        ext_words = words
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)
        feat_embed = []
        if 'tag' in self.args.feat:
            feat_embed.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embed.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embed.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embed.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embed.append(self.lemma_embed(feats.pop(0)))
        if isinstance(self.embed_dropout, IndependentDropout):
            if len(feat_embed) == 0:
                raise RuntimeError(f'`feat` is not allowed to be empty, which is {self.args.feat} now')
            embed = torch.cat(self.embed_dropout(word_embed, torch.cat(feat_embed, -1)), -1)
        else:
            embed = word_embed
            if len(feat_embed) > 0:
                embed = torch.cat((embed, torch.cat(feat_embed, -1)), -1)
            embed = self.embed_dropout(embed)
        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        elif self.args.encoder == 'transformer':
            x = self.encoder(self.embed(words, feats), words.ne(self.args.pad_index))
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError


def debinarize(fbin: str, pos_or_key: Optional[Union[Tuple[int, int], str]]=(0, 0), meta: bool=False) ->Union[Any, Iterable[Any]]:
    with open(fbin, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if meta or isinstance(pos_or_key, str):
            length = len(struct.pack('LL', 0, 0))
            mm.seek(-length, os.SEEK_END)
            offset, length = struct.unpack('LL', mm.read(length))
            mm.seek(offset)
            if meta:
                return pickle.loads(mm.read(length))
            objs, meta = [], pickle.loads(mm.read(length))[pos_or_key]
            for offset, length in meta.tolist():
                mm.seek(offset)
                objs.append(pickle.loads(mm.read(length)))
            return objs
        offset, length = pos_or_key
        mm.seek(offset)
        return pickle.loads(mm.read(length))


class DataLoader(torch.utils.data.DataLoader):
    """
    A wrapper for native :class:`torch.utils.data.DataLoader` enhanced with a data prefetcher.
    See http://stackoverflow.com/questions/7323664/python-generator-pre-fetch and
    https://github.com/NVIDIA/apex/issues/304.
    """

    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def __iter__(self):
        return PrefetchGenerator(self.transform, super().__iter__())


INF = float('inf')


def binarize(data: Union[List[str], Dict[str, Iterable]], fbin: str=None, merge: bool=False) ->Tuple[str, torch.Tensor]:
    start, meta = 0, defaultdict(list)
    with open(fbin, 'wb') as f:
        if merge:
            for file in data:
                if not os.path.exists(file):
                    raise RuntimeError('Some files are missing. Please check the paths')
                mi = debinarize(file, meta=True)
                for key, val in mi.items():
                    val[:, 0] += start
                    meta[key].append(val)
                with open(file, 'rb') as fi:
                    length = int(sum(val[:, 1].sum() for val in mi.values()))
                    f.write(fi.read(length))
                start = start + length
            meta = {key: torch.cat(val) for key, val in meta.items()}
        else:
            for key, val in data.items():
                for i in val:
                    bytes = pickle.dumps(i)
                    f.write(bytes)
                    meta[key].append((start, len(bytes)))
                    start = start + len(bytes)
            meta = {key: torch.tensor(val) for key, val in meta.items()}
        pickled = pickle.dumps(meta)
        f.write(pickled)
        f.write(struct.pack('LL', start, len(pickled)))
    return fbin, meta


def collate_fn(x):
    return Batch(x)


def kmeans(x: List[int], k: int, max_it: int=32) ->Tuple[List[float], List[List[int]]]:
    """
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (List[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters, which is an approximate value.
            The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        List[float], List[List[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10, 20, (10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """
    x = torch.tensor(x, dtype=torch.float)
    datapoints, indices, freqs = x.unique(return_inverse=True, return_counts=True)
    k = min(len(datapoints), k)
    centroids = datapoints[torch.randperm(len(datapoints))[:k]]
    dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)
    for _ in range(max_it):
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        for i in none:
            biggest = torch.where(mask[mask.sum(-1).argmax()])[0]
            farthest = dists[biggest].argmax()
            y[biggest[farthest]] = i
            mask = torch.arange(k).unsqueeze(-1).eq(y)
        centroids, old = (datapoints * freqs * mask).sum(-1) / (freqs * mask).sum(-1), centroids
        dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)
        if centroids.equal(old):
            break
    assigned = y.unique().tolist()
    centroids = centroids[assigned].tolist()
    clusters = [torch.where(indices.unsqueeze(-1).eq(torch.where(y.eq(i))[0]).any(-1))[0].tolist() for i in assigned]
    return centroids, clusters


NUL = '<nul>'


class AttachJuxtaposeConstituencyModel(Model):
    """
    The implementation of AttachJuxtapose Constituency Parser :cite:`yang-deng-2020-aj`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layers. Default: .33.
        n_gnn_layers (int):
            The number of GNN layers. Default: 3.
        gnn_dropout (float):
            The dropout ratio of GNN layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, True), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_gnn_layers=3, gnn_dropout=0.33, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.label_embed = nn.Embedding(n_labels + 1, self.args.n_encoder_hidden)
        self.gnn_layers = GraphConvolutionalNetwork(n_model=self.args.n_encoder_hidden, n_layers=self.args.n_gnn_layers, dropout=self.args.gnn_dropout)
        self.node_classifier = nn.Sequential(nn.Linear(2 * self.args.n_encoder_hidden, self.args.n_encoder_hidden // 2), nn.LayerNorm(self.args.n_encoder_hidden // 2), nn.ReLU(), nn.Linear(self.args.n_encoder_hidden // 2, 1))
        self.label_classifier = nn.Sequential(nn.Linear(2 * self.args.n_encoder_hidden, self.args.n_encoder_hidden // 2), nn.LayerNorm(self.args.n_encoder_hidden // 2), nn.ReLU(), nn.Linear(self.args.n_encoder_hidden // 2, 2 * n_labels))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: torch.LongTensor, feats: List[torch.LongTensor]=None) ->torch.Tensor:
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor:
                Contextualized output hidden states of shape ``[batch_size, seq_len, n_model]`` of the input.
        """
        return self.encode(words, feats)

    def loss(self, x: torch.Tensor, nodes: torch.LongTensor, parents: torch.LongTensor, news: torch.LongTensor, mask: torch.BoolTensor) ->torch.Tensor:
        """
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, n_model]``.
                Contextualized output hidden states.
            nodes (~torch.LongTensor): ``[batch_size, seq_len]``.
                The target node positions on rightmost chains.
            parents (~torch.LongTensor): ``[batch_size, seq_len]``.
                The parent node labels of terminals.
            news (~torch.LongTensor): ``[batch_size, seq_len]``.
                The parent node labels of juxtaposed targets and terminals.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        spans, s_node, x_node = None, [], []
        actions = torch.stack((nodes, parents, news))
        for t, action in enumerate(actions.unbind(-1)):
            if t == 0:
                x_span = self.label_embed(actions.new_full((x.shape[0], 1), self.args.n_labels))
                span_mask = mask[:, :1]
            else:
                x_span = self.rightmost_chain(x, spans, mask, t)
                span_lens = spans[:, :-1, -1].ge(0).sum(-1)
                span_mask = span_lens.unsqueeze(-1).gt(x.new_tensor(range(span_lens.max())))
            x_rightmost = torch.cat((x_span, x[:, t].unsqueeze(1).expand_as(x_span)), -1)
            s_node.append(self.node_classifier(x_rightmost).squeeze(-1))
            s_node[-1] = s_node[-1].masked_fill_(~span_mask, -INF).masked_fill(~span_mask.any(-1).unsqueeze(-1), 0)
            x_node.append(torch.bmm(s_node[-1].softmax(-1).unsqueeze(1), x_span).squeeze(1))
            spans = AttachJuxtaposeTree.action2span(action, spans, self.args.nul_index, mask[:, t])
        attach_mask = x.new_tensor(range(self.args.n_labels)).eq(self.args.nul_index)
        s_node, x_node = pad(s_node, -INF).transpose(0, 1), torch.stack(x_node, 1)
        s_parent, s_new = self.label_classifier(torch.cat((x, x_node), -1)).chunk(2, -1)
        s_parent = torch.cat((s_parent[:, :1].masked_fill(attach_mask, -INF), s_parent[:, 1:]), 1)
        s_new = torch.cat((s_new[:, :1].masked_fill(~attach_mask, -INF), s_new[:, 1:]), 1)
        node_loss = self.criterion(s_node[mask], nodes[mask])
        label_loss = self.criterion(s_parent[mask], parents[mask]) + self.criterion(s_new[mask], news[mask])
        return node_loss + label_loss

    def decode(self, x: torch.Tensor, mask: torch.BoolTensor, beam_size: int=1) ->List[List[Tuple]]:
        """
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, n_model]``.
                Contextualized output hidden states.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            beam_size (int):
                Beam size for decoding. Default: 1.

        Returns:
            List[List[Tuple]]:
                Sequences of factorized labeled trees.
        """
        spans = None
        batch_size, *_ = x.shape
        n_labels = self.args.n_labels
        x = x.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *x.shape[1:])
        mask = mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, *mask.shape[1:])
        batches = x.new_tensor(range(batch_size)).long() * beam_size
        scores = x.new_full((batch_size, beam_size), -INF).index_fill_(-1, x.new_tensor(0).long(), 0).view(-1)
        for t in range(x.shape[1]):
            if t == 0:
                x_span = self.label_embed(batches.new_full((x.shape[0], 1), n_labels))
                span_mask = mask[:, :1]
            else:
                x_span = self.rightmost_chain(x, spans, mask, t)
                span_lens = spans[:, :-1, -1].ge(0).sum(-1)
                span_mask = span_lens.unsqueeze(-1).gt(x.new_tensor(range(span_lens.max())))
            s_node = self.node_classifier(torch.cat((x_span, x[:, t].unsqueeze(1).expand_as(x_span)), -1)).squeeze(-1)
            s_node = s_node.masked_fill_(~span_mask, -INF).masked_fill(~span_mask.any(-1).unsqueeze(-1), 0).log_softmax(-1)
            x_node = torch.bmm(s_node.exp().unsqueeze(1), x_span).squeeze(1)
            s_parent, s_new = self.label_classifier(torch.cat((x[:, t], x_node), -1)).chunk(2, -1)
            s_parent, s_new = s_parent.log_softmax(-1), s_new.log_softmax(-1)
            if t == 0:
                s_parent[:, self.args.nul_index] = -INF
                s_new[:, s_new.new_tensor(range(self.args.n_labels)).ne(self.args.nul_index)] = -INF
            s_node, nodes = s_node.topk(min(s_node.shape[-1], beam_size), -1)
            s_parent, parents = s_parent.topk(min(n_labels, beam_size), -1)
            s_new, news = s_new.topk(min(n_labels, beam_size), -1)
            s_action = s_node.unsqueeze(2) + (s_parent.unsqueeze(2) + s_new.unsqueeze(1)).view(x.shape[0], 1, -1)
            s_action = s_action.view(x.shape[0], -1)
            k_beam, k_node, k_parent = s_action.shape[-1], parents.shape[-1] * news.shape[-1], news.shape[-1]
            scores = scores.unsqueeze(-1) + s_action
            scores, cands = scores.view(batch_size, -1).topk(beam_size, -1)
            scores = scores.view(-1)
            beams = cands.div(k_beam, rounding_mode='floor')
            nodes = nodes.view(batch_size, -1).gather(-1, cands.div(k_node, rounding_mode='floor'))
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            parents = parents[indices].view(batch_size, -1).gather(-1, cands.div(k_parent, rounding_mode='floor') % k_parent)
            news = news[indices].view(batch_size, -1).gather(-1, cands % k_parent)
            action = torch.stack((nodes, parents, news)).view(3, -1)
            spans = spans[indices] if spans is not None else None
            spans = AttachJuxtaposeTree.action2span(action, spans, self.args.nul_index, mask[:, t])
        mask = mask.view(batch_size, beam_size, -1)[:, 0]
        spans = spans[batches + scores.view(batch_size, -1).argmax(-1)]
        span_mask = spans.ge(0)
        span_indices = torch.where(span_mask)
        span_labels = spans[span_indices]
        chart_preds = [[] for _ in range(x.shape[0])]
        for i, *span in zip(*[s.tolist() for s in span_indices], span_labels.tolist()):
            chart_preds[i].append(span)
        return chart_preds

    def rightmost_chain(self, x: torch.Tensor, spans: torch.LongTensor, mask: torch.BoolTensor, t: int) ->torch.Tensor:
        x_p, mask_p = x[:, :t], mask[:, :t]
        lens = mask_p.sum(-1)
        span_mask = spans[:, :-1, 1:].ge(0)
        span_lens = span_mask.sum((-1, -2))
        span_indices = torch.where(span_mask)
        span_labels = spans[:, :-1, 1:][span_indices]
        x_span = self.label_embed(span_labels)
        x_span += x[span_indices[0], span_indices[1]] + x[span_indices[0], span_indices[2]]
        node_lens = lens + span_lens
        adj_mask = node_lens.unsqueeze(-1).gt(x.new_tensor(range(node_lens.max())))
        x_mask = lens.unsqueeze(-1).gt(x.new_tensor(range(adj_mask.shape[-1])))
        span_mask = ~x_mask & adj_mask
        x_tree = x.new_zeros(*adj_mask.shape, x.shape[-1]).masked_scatter_(x_mask.unsqueeze(-1), x_p[mask_p])
        x_tree = x_tree.masked_scatter_(span_mask.unsqueeze(-1), x_span)
        adj = mask.new_zeros(*x_tree.shape[:-1], x_tree.shape[1])
        adj_spans = lens.new_tensor(range(x_tree.shape[1])).view(1, 1, -1).repeat(2, x.shape[0], 1)
        adj_spans = adj_spans.masked_scatter_(span_mask.unsqueeze(0), torch.stack(span_indices[1:]))
        adj_l, adj_r, adj_w = *adj_spans.unbind(), adj_spans[1] - adj_spans[0]
        adj_parent = adj_l.unsqueeze(-1).ge(adj_l.unsqueeze(-2)) & adj_r.unsqueeze(-1).le(adj_r.unsqueeze(-2))
        adj_parent.diagonal(0, 1, 2).copy_(adj_w.eq(t - 1))
        adj_parent = adj_parent & span_mask.unsqueeze(1)
        adj_parent = (adj_w.unsqueeze(-2) - adj_w.unsqueeze(-1)).masked_fill_(~adj_parent, t).argmin(-1)
        adj.scatter_(-1, adj_parent.unsqueeze(-1), 1)
        adj = (adj | adj.transpose(-1, -2)).float()
        x_tree = self.gnn_layers(x_tree, adj, adj_mask)
        span_mask = span_mask.masked_scatter(span_mask, span_indices[2].eq(t - 1))
        span_lens = span_mask.sum(-1)
        x_tree, span_mask = x_tree[span_mask], span_lens.unsqueeze(-1).gt(x.new_tensor(range(span_lens.max())))
        x_span = x.new_zeros(*span_mask.shape, x.shape[-1]).masked_scatter_(span_mask.unsqueeze(-1), x_tree)
        return x_span


MIN = -1e+32


class Semiring(object):
    """
    Base semiring class :cite:`goodman-1999-semiring`.

    A semiring is defined by a tuple :math:`<K, \\oplus, \\otimes, \\mathbf{0}, \\mathbf{1}>`.
    :math:`K` is a set of values;
    :math:`\\oplus` is commutative, associative and has an identity element `0`;
    :math:`\\otimes` is associative, has an identity element `1` and distributes over `+`.
    """
    zero = 0
    one = 1

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return x + y

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return x * y

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.sum(dim)

    @classmethod
    def prod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.prod(dim)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.cumsum(dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.cumprod(dim)

    @classmethod
    def dot(cls, x: torch.Tensor, y: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return cls.sum(cls.mul(x, y), dim)

    @classmethod
    def times(cls, *x: Iterable[torch.Tensor]) ->torch.Tensor:
        return reduce(lambda i, j: cls.mul(i, j), x)

    @classmethod
    def zero_(cls, x: torch.Tensor) ->torch.Tensor:
        return x.fill_(cls.zero)

    @classmethod
    def one_(cls, x: torch.Tensor) ->torch.Tensor:
        return x.fill_(cls.one)

    @classmethod
    def zero_mask(cls, x: torch.Tensor, mask: torch.BoolTensor) ->torch.Tensor:
        return x.masked_fill(mask, cls.zero)

    @classmethod
    def zero_mask_(cls, x: torch.Tensor, mask: torch.BoolTensor) ->torch.Tensor:
        return x.masked_fill_(mask, cls.zero)

    @classmethod
    def one_mask(cls, x: torch.Tensor, mask: torch.BoolTensor) ->torch.Tensor:
        return x.masked_fill(mask, cls.one)

    @classmethod
    def one_mask_(cls, x: torch.Tensor, mask: torch.BoolTensor) ->torch.Tensor:
        return x.masked_fill_(mask, cls.one)

    @classmethod
    def zeros_like(cls, x: torch.Tensor) ->torch.Tensor:
        return x.new_full(x.shape, cls.zero)

    @classmethod
    def ones_like(cls, x: torch.Tensor) ->torch.Tensor:
        return x.new_full(x.shape, cls.one)

    @classmethod
    def convert(cls, x: torch.Tensor) ->torch.Tensor:
        return x

    @classmethod
    def unconvert(cls, x: torch.Tensor) ->torch.Tensor:
        return x


class LogSemiring(Semiring):
    """
    Log-space semiring :math:`<\\mathrm{logsumexp}, +, -\\infty, 0>`.
    """
    zero = MIN
    one = 0

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return x.logaddexp(y)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return x + y

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.logsumexp(dim)

    @classmethod
    def prod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.sum(dim)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.logcumsumexp(dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.cumsum(dim)


class CrossEntropySemiring(LogSemiring):
    """
    Cross Entropy expectation semiring :math:`<\\oplus, +, [-\\infty, -\\infty, 0], [0, 0, 0]>`,
    where :math:`\\oplus` computes the log-values and the running distributional cross entropy :math:`H[p,q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return cls.sum(torch.stack((x, y)), 0)

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul(x[..., -1] - r[..., 1]).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.add(x, y))), dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.mul(x, y))), dim)

    @classmethod
    def zero_(cls, x: torch.Tensor) ->torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) ->torch.Tensor:
        return x.fill_(cls.one)

    @classmethod
    def convert(cls, x: torch.Tensor) ->torch.Tensor:
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) ->torch.Tensor:
        return x[..., -1]


class EntropySemiring(LogSemiring):
    """
    Entropy expectation semiring :math:`<\\oplus, +, [-\\infty, 0], [0, 0]>`,
    where :math:`\\oplus` computes the log-values and the running distributional entropy :math:`H[p]`
    :cite:`li-eisner-2009-first,hwa-2000-sample,kim-etal-2019-unsupervised`.
    """

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return cls.sum(torch.stack((x, y)), 0)

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        p = x[..., 0].logsumexp(dim)
        r = x[..., 0] - p.unsqueeze(dim)
        r = r.exp().mul(x[..., -1] - r).sum(dim)
        return torch.stack((p, r), -1)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.add(x, y))), dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.mul(x, y))), dim)

    @classmethod
    def zero_(cls, x: torch.Tensor) ->torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) ->torch.Tensor:
        return x.fill_(cls.one)

    @classmethod
    def convert(cls, x: torch.Tensor) ->torch.Tensor:
        return torch.stack((x, cls.ones_like(x)), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) ->torch.Tensor:
        return x[..., -1]


class KLDivergenceSemiring(LogSemiring):
    """
    KL divergence expectation semiring :math:`<\\oplus, +, [-\\infty, -\\infty, 0], [0, 0, 0]>`,
    where :math:`\\oplus` computes the log-values and the running distributional KL divergence :math:`KL[p \\parallel q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return cls.sum(torch.stack((x, y)), 0)

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul(x[..., -1] - r[..., 1] + r[..., 0]).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.add(x, y))), dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.mul(x, y))), dim)

    @classmethod
    def zero_(cls, x: torch.Tensor) ->torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) ->torch.Tensor:
        return x.fill_(cls.one)

    @classmethod
    def convert(cls, x: torch.Tensor) ->torch.Tensor:
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) ->torch.Tensor:
        return x[..., -1]


def KMaxSemiring(k):
    """
    k-max semiring :math:`<\\mathrm{kmax}, +, [-\\infty, -\\infty, \\dots], [0, -\\infty, \\dots]>`.
    """


    class KMaxSemiring(LogSemiring):

        @classmethod
        def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
            return x.unsqueeze(-1).max(y.unsqueeze(-2)).flatten(-2).topk(k, -1)[0]

        @classmethod
        def mul(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
            return (x.unsqueeze(-1) + y.unsqueeze(-2)).flatten(-2).topk(k, -1)[0]

        @classmethod
        def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
            return x.movedim(dim, -1).flatten(-2).topk(k, -1)[0]

        @classmethod
        def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
            return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.add(x, y))), dim)

        @classmethod
        def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
            return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.mul(x, y))), dim)

        @classmethod
        def one_(cls, x: torch.Tensor) ->torch.Tensor:
            x[..., :1].fill_(cls.one)
            x[..., 1:].fill_(cls.zero)
            return x

        @classmethod
        def convert(cls, x: torch.Tensor) ->torch.Tensor:
            return torch.cat((x.unsqueeze(-1), cls.zero_(x.new_empty(*x.shape, k - 1))), -1)
    return KMaxSemiring


class MaxSemiring(LogSemiring):
    """
    Max semiring :math:`<\\mathrm{max}, +, -\\infty, 0>`.
    """

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return x.max(y)

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.max(dim)[0]

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return x.cummax(dim)


class SampledSemiring(LogSemiring):
    """
    Sampling semiring :math:`<\\mathrm{logsumexp}, +, -\\infty, 0>`,
    which is an exact forward-filtering, backward-sampling approach.
    """

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return cls.sum(torch.stack((x, y)), 0)

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return sampled_logsumexp(x, dim)

    @classmethod
    def cumsum(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.add(x, y))), dim)

    @classmethod
    def cumprod(cls, x: torch.Tensor, dim: int=-1) ->torch.Tensor:
        return torch.stack(list(itertools.accumulate(x.unbind(dim), lambda x, y: cls.mul(x, y))), dim)


def stripe(x: torch.Tensor, n: int, w: int, offset: Tuple=(0, 0), horizontal: bool=True) ->torch.Tensor:
    """
    Returns a parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        horizontal (bool): `True` if returns a horizontal stripe; `False` otherwise.

    Returns:
        A parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """
    x = x.contiguous()
    seq_len, stride = x.size(1), list(x.stride())
    numel = stride[1]
    return x.as_strided(size=(n, w, *x.shape[2:]), stride=[(seq_len + 1) * numel, (1 if horizontal else seq_len) * numel] + stride[2:], storage_offset=(offset[0] * seq_len + offset[1]) * numel)


class CRFConstituencyModel(Model):
    """
    The implementation of CRF Constituency Parser :cite:`zhang-etal-2020-fast`,
    also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_span_mlp (int):
            Span MLP size. Default: 500.
        n_label_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, True), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_span_mlp=500, n_label_mlp=100, mlp_dropout=0.33, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.span_mlp_l = MLP(n_in=self.args.n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.span_mlp_r = MLP(n_in=self.args.n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.label_mlp_l = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.label_mlp_r = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.span_attn = Biaffine(n_in=n_span_mlp, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """
        x = self.encode(words, feats)
        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        span_l = self.span_mlp_l(x)
        span_r = self.span_mlp_r(x)
        label_l = self.label_mlp_l(x)
        label_r = self.label_mlp_r(x)
        s_span = self.span_attn(span_l, span_r)
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)
        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        """
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all constituent labels.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels. Positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and original constituent scores
                of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        span_mask = charts.ge(0) & mask
        span_dist = ConstituencyCRF(s_span, mask[:, 0].sum(-1))
        span_loss = -span_dist.log_prob(charts).sum() / mask[:, 0].sum()
        span_probs = span_dist.marginals if mbr else s_span
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss
        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        """
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all constituent labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            List[List[Tuple]]:
                Sequences of factorized labeled trees.
        """
        span_preds = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).argmax
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]


class VIConstituencyModel(CRFConstituencyModel):
    """
    The implementation of Constituency Parser using variational inference.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_span_mlp (int):
            Span MLP size. Default: 500.
        n_pair_mlp (int):
            Binary factor MLP size. Default: 100.
        n_label_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: ``mfvi``.
        max_iter (int):
            Max iteration times for inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, True), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_span_mlp=500, n_pair_mlp=100, n_label_mlp=100, mlp_dropout=0.33, inference='mfvi', max_iter=3, interpolation=0.1, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.span_mlp_l = MLP(n_in=self.args.n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.span_mlp_r = MLP(n_in=self.args.n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.pair_mlp_l = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=mlp_dropout)
        self.pair_mlp_r = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=mlp_dropout)
        self.pair_mlp_b = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=mlp_dropout)
        self.label_mlp_l = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.label_mlp_r = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.span_attn = Biaffine(n_in=n_span_mlp, bias_x=True, bias_y=False)
        self.pair_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.inference = (ConstituencyMFVI if inference == 'mfvi' else ConstituencyLBP)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible constituents (``[batch_size, seq_len, seq_len]``),
                second-order triples (``[batch_size, seq_len, seq_len, n_labels]``) and
                all possible labels on each constituent (``[batch_size, seq_len, seq_len, n_labels]``).
        """
        x = self.encode(words, feats)
        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        span_l = self.span_mlp_l(x)
        span_r = self.span_mlp_r(x)
        pair_l = self.pair_mlp_l(x)
        pair_r = self.pair_mlp_r(x)
        pair_b = self.pair_mlp_b(x)
        label_l = self.label_mlp_l(x)
        label_r = self.label_mlp_r(x)
        s_span = self.span_attn(span_l, span_r)
        s_pair = self.pair_attn(pair_l, pair_r, pair_b).permute(0, 3, 1, 2)
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)
        return s_span, s_pair, s_label

    def loss(self, s_span, s_pair, s_label, charts, mask):
        """
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_pair (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of second-order triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all constituent labels.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels. Positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and marginals of shape ``[batch_size, seq_len, seq_len]``.
        """
        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.inference((s_span, s_pair), mask, span_mask)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = self.args.interpolation * label_loss + (1 - self.args.interpolation) * span_loss
        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        """
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all constituents.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all constituent labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            List[List[Tuple]]:
                Sequences of factorized labeled trees.
        """
        span_preds = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).argmax
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]


def tarjan(sequence: Iterable[int]) ->Iterable[int]:
    """
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices making up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """
    sequence = [-1] + sequence
    dfn = [-1] * len(sequence)
    low = [-1] * len(sequence)
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True
        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            if len(cycle) > 1:
                yield cycle
    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s: torch.Tensor) ->torch.Tensor:
    """
    ChuLiu/Edmonds algorithm for non-projective decoding :cite:`mcdonald-etal-2005-non`.

    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in :cite:`mcdonald-etal-2005-non`.

    Notes:
        The algorithm does not guarantee to parse a single-root tree.

    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.

    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    """
    s[0, 1:] = MIN
    s.diagonal()[1:].fill_(MIN)
    tree = s.argmax(-1)
    cycle = next(tarjan(tree.tolist()[1:]), None)
    if not cycle:
        return tree
    cycle = torch.tensor(cycle)
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        cycle_heads = tree[cycle]
        s_cycle = s[cycle, cycle_heads]
        s_dep = s[noncycle][:, cycle]
        deps = s_dep.argmax(1)
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        heads = s_head.argmax(0)
        contracted = torch.cat((noncycle, torch.tensor([-1])))
        s = s[contracted][:, contracted]
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        s[-1, :-1] = s_head[heads, range(len(heads))]
        return s, heads, deps
    s, heads, deps = contract(s)
    y = chuliu_edmonds(s)
    y, cycle_head = y[:-1], y[-1]
    subtree = y < len(y)
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    subtree = ~subtree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    cycle_root = heads[cycle_head]
    tree[cycle[cycle_root]] = noncycle[cycle_head]
    return tree


def mst(scores: torch.Tensor, mask: torch.BoolTensor, multiroot: bool=False) ->torch.Tensor:
    """
    MST algorithm for decoding non-projective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.

    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = MIN
        >>> scores.diagonal(0, 1, 2)[1:].fill_(MIN)
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """
    _, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()
    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length + 1, :length + 1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = MIN
            s = s.index_fill(1, torch.tensor(0), MIN)
            for root in roots:
                s[:, 0] = MIN
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)
    return pad(preds, total_length=seq_len)


class BiaffineDependencyModel(Model):
    """
    The implementation of Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_rels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, False), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_arc_mlp=500, n_rel_mlp=100, mlp_dropout=0.33, scale=0, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """
        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [(not CoNLL.istree(seq[1:i + 1], proj)) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds


class CRFDependencyModel(BiaffineDependencyModel):
    """
    The implementation of first-order CRF Dependency Parser
    :cite:`zhang-etal-2020-efficient,ma-hovy-2017-neural,koo-etal-2007-structured`).

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
        proj (bool):
            If ``True``, takes :class:`DependencyCRF` as inference layer, :class:`MatrixTree` otherwise.
            Default: ``True``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=True, partial=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        CRF = DependencyCRF if self.args.proj else MatrixTree
        arc_dist = CRF(s_arc, mask.sum(-1))
        arc_loss = -arc_dist.log_prob(arcs, partial=partial).sum() / mask.sum()
        arc_probs = arc_dist.marginals if mbr else s_arc
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs


class CRF2oDependencyModel(BiaffineDependencyModel):
    """
    The implementation of second-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_sib_mlp (int):
            Sibling MLP size. Default: 100.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_rels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, False), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_arc_mlp=500, n_sib_mlp=100, n_rel_mlp=100, mlp_dropout=0.33, scale=0, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.sib_mlp_s = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=n_sib_mlp, scale=scale, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible arcs (``[batch_size, seq_len, seq_len]``),
                dependent-head-sibling triples (``[batch_size, seq_len, seq_len, seq_len]``) and
                all possible labels on each arc (``[batch_size, seq_len, seq_len, n_labels]``).
        """
        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        sib_s = self.sib_mlp_s(x)
        sib_d = self.sib_mlp_d(x)
        sib_h = self.sib_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask, mbr=True, partial=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            sibs (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard siblings.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        arc_dist = Dependency2oCRF((s_arc, s_sib), mask.sum(-1))
        arc_loss = -arc_dist.log_prob((arcs, sibs), partial=partial).sum() / mask.sum()
        if mbr:
            s_arc, s_sib = arc_dist.marginals
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, s_arc, s_sib

    def decode(self, s_arc, s_sib, s_rel, mask, tree=False, mbr=True, proj=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [(not CoNLL.istree(seq[1:i + 1], proj)) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            if proj:
                arc_preds[bad] = Dependency2oCRF((s_arc[bad], s_sib[bad]), mask[bad].sum(-1)).argmax
            else:
                arc_preds[bad] = MatrixTree(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds


class VIDependencyModel(BiaffineDependencyModel):
    """
    The implementation of Dependency Parser using Variational Inference :cite:`wang-tu-2020-second`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_sib_mlp (int):
            Binary factor MLP size. Default: 100.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        inference (str):
            Approximate inference methods. Default: ``mfvi``.
        max_iter (int):
            Max iteration times for inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_rels, n_tags=None, n_chars=None, encoder='lstm', feat=['char'], n_embed=100, n_pretrained=100, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, elmo='original_5b', elmo_bos_eos=(True, False), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.33, n_encoder_hidden=800, n_encoder_layers=3, encoder_dropout=0.33, n_arc_mlp=500, n_sib_mlp=100, n_rel_mlp=100, mlp_dropout=0.33, scale=0, inference='mfvi', max_iter=3, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.sib_mlp_s = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=n_sib_mlp, scale=scale, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.inference = (DependencyMFVI if inference == 'mfvi' else DependencyLBP)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible arcs (``[batch_size, seq_len, seq_len]``),
                dependent-head-sibling triples (``[batch_size, seq_len, seq_len, seq_len]``) and
                all possible labels on each arc (``[batch_size, seq_len, seq_len, n_labels]``).
        """
        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        sib_s = self.sib_mlp_s(x)
        sib_d = self.sib_mlp_d(x)
        sib_h = self.sib_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, rels, mask):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        arc_loss, marginals = self.inference((s_arc, s_sib), mask, arcs)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, marginals

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [(not CoNLL.istree(seq[1:i + 1], proj)) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds


class BiaffineSemanticDependencyModel(Model):
    """
    The implementation of Biaffine Semantic Dependency Parser :cite:`dozat-manning-2018-simpler`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word representations. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 1200.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Edge MLP size. Default: 600.
        n_label_mlp  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None, n_lemmas=None, encoder='lstm', feat=['tag', 'char', 'lemma'], n_embed=100, n_pretrained=125, n_feat_embed=100, n_char_embed=50, n_char_hidden=400, char_pad_index=0, char_dropout=0.33, elmo='original_5b', elmo_bos_eos=(True, False), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.2, n_encoder_hidden=1200, n_encoder_layers=3, encoder_dropout=0.33, n_edge_mlp=600, n_label_mlp=600, edge_mlp_dropout=0.25, label_mlp_dropout=0.33, interpolation=0.1, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.edge_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.edge_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.label_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.label_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.edge_attn = Biaffine(n_in=n_edge_mlp, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
        return self

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """
        x = self.encode(words, feats)
        edge_d = self.edge_mlp_d(x)
        edge_h = self.edge_mlp_h(x)
        label_d = self.label_mlp_d(x)
        label_h = self.label_mlp_h(x)
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)
        return s_edge, s_label

    def loss(self, s_edge, s_label, labels, mask):
        """
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        return self.args.interpolation * label_loss + (1 - self.args.interpolation) * edge_loss

    def decode(self, s_edge, s_label):
        """
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """
        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1)


class VISemanticDependencyModel(BiaffineSemanticDependencyModel):
    """
    The implementation of Semantic Dependency Parser using Variational Inference :cite:`wang-etal-2019-second`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 1200.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Unary factor MLP size. Default: 600.
        n_pair_mlp (int):
            Binary factor MLP size. Default: 150.
        n_label_mlp  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .25.
        pair_mlp_dropout (float):
            The dropout ratio of binary factor MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: ``mfvi``.
        max_iter (int):
            Max iteration times for inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, n_words, n_labels, n_tags=None, n_chars=None, n_lemmas=None, encoder='lstm', feat=['tag', 'char', 'lemma'], n_embed=100, n_pretrained=125, n_feat_embed=100, n_char_embed=50, n_char_hidden=100, char_pad_index=0, char_dropout=0, elmo='original_5b', elmo_bos_eos=(True, False), bert=None, n_bert_layers=4, mix_dropout=0.0, bert_pooling='mean', bert_pad_index=0, finetune=False, n_plm_embed=0, embed_dropout=0.2, n_encoder_hidden=1200, n_encoder_layers=3, encoder_dropout=0.33, n_edge_mlp=600, n_pair_mlp=150, n_label_mlp=600, edge_mlp_dropout=0.25, pair_mlp_dropout=0.25, label_mlp_dropout=0.33, inference='mfvi', max_iter=3, interpolation=0.1, pad_index=0, unk_index=1, **kwargs):
        super().__init__(**Config().update(locals()))
        self.edge_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.edge_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.pair_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.pair_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.pair_mlp_g = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.label_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.label_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.edge_attn = Biaffine(n_in=n_edge_mlp, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.inference = (SemanticDependencyMFVI if inference == 'mfvi' else SemanticDependencyLBP)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        """
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                The first and last are scores of all possible edges of shape ``[batch_size, seq_len, seq_len]``
                and possible labels on each edge of shape ``[batch_size, seq_len, seq_len, n_labels]``.
                Others are scores of second-order sibling, coparent and grandparent factors
                (``[batch_size, seq_len, seq_len, seq_len]``).

        """
        x = self.encode(words, feats)
        edge_d = self.edge_mlp_d(x)
        edge_h = self.edge_mlp_h(x)
        pair_d = self.pair_mlp_d(x)
        pair_h = self.pair_mlp_h(x)
        pair_g = self.pair_mlp_g(x)
        label_d = self.label_mlp_d(x)
        label_h = self.label_mlp_h(x)
        s_edge = self.edge_attn(edge_d, edge_h)
        s_sib = self.sib_attn(pair_d, pair_d, pair_h)
        s_sib = (s_sib.triu() + s_sib.triu(1).transpose(-1, -2)).permute(0, 3, 1, 2)
        s_cop = self.cop_attn(pair_h, pair_d, pair_h).permute(0, 3, 1, 2)
        s_cop = s_cop.triu() + s_cop.triu(1).transpose(-1, -2)
        s_grd = self.grd_attn(pair_g, pair_d, pair_h).permute(0, 3, 1, 2)
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)
        return s_edge, s_sib, s_cop, s_grd, s_label

    def loss(self, s_edge, s_sib, s_cop, s_grd, s_label, labels, mask):
        """
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_cop (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-coparent triples.
            s_grd (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-grandparent triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and marginals of shape ``[batch_size, seq_len, seq_len]``.
        """
        edge_mask = labels.ge(0) & mask
        edge_loss, marginals = self.inference((s_edge, s_sib, s_cop, s_grd), mask, edge_mask.long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        loss = self.args.interpolation * label_loss + (1 - self.args.interpolation) * edge_loss
        return loss, marginals

    def decode(self, s_edge, s_label):
        """
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """
        return s_label.argmax(-1).masked_fill_(s_edge.lt(0.5), -1)


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SinusoidPositionalEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SinusoidRelativePositionalEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_yzhangcs_parser(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

