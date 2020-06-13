import sys
_module = sys.modules[__name__]
del sys
bert_pytorch = _module
__main__ = _module
dataset = _module
vocab = _module
model = _module
attention = _module
multi_head = _module
single = _module
bert = _module
embedding = _module
bert = _module
position = _module
segment = _module
token = _module
language_model = _module
transformer = _module
utils = _module
feed_forward = _module
gelu = _module
layer_norm = _module
sublayer = _module
trainer = _module
optim_schedule = _module
pretrain = _module
setup = _module
test = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


import math


from torch.optim import Adam


from torch.utils.data import DataLoader


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for
            _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).
            transpose(1, 2) for l, x in zip(self.linear_layers, (query, key,
            value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self
            .dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h *
            self.d_k)
        return self.output_linear(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query
            .size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12,
        dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden
            )
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden,
            attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=
            embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(
            segment_label)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(
            10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):

    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, (0)]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=
            feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x,
            _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_codertimo_BERT_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(MultiHeadedAttention(*[], **{'h': 4, 'd_model': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Attention(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BERT(*[], **{'vocab_size': 4}), [torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

    def test_003(self):
        self._check(BERTEmbedding(*[], **{'vocab_size': 4, 'embed_size': 4}), [torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

    def test_004(self):
        self._check(PositionalEmbedding(*[], **{'d_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SegmentEmbedding(*[], **{}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_006(self):
        self._check(TokenEmbedding(*[], **{'vocab_size': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_007(self):
        self._check(NextSentencePrediction(*[], **{'hidden': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(MaskedLanguageModel(*[], **{'hidden': 4, 'vocab_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(TransformerBlock(*[], **{'hidden': 4, 'attn_heads': 4, 'feed_forward_hidden': 4, 'dropout': 0.5}), [torch.rand([4, 1]), torch.rand([4, 1])], {})

    def test_010(self):
        self._check(PositionwiseFeedForward(*[], **{'d_model': 4, 'd_ff': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(GELU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(LayerNorm(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(SublayerConnection(*[], **{'size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), ReLU()], {})

