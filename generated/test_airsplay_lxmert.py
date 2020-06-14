import sys
_module = sys.modules[__name__]
del sys
process_data = _module
process_data_all = _module
process_submit_data = _module
extract_coco_image = _module
process_dataset = _module
extract_nlvr2_image = _module
extract_gqa_image = _module
entry = _module
file_utils = _module
modeling = _module
optimization = _module
tokenization = _module
param = _module
lxmert_data = _module
lxmert_pretrain = _module
qa_answer_table = _module
gqa = _module
gqa_data = _module
gqa_model = _module
nlvr2 = _module
nlvr2_data = _module
nlvr2_model = _module
vqa = _module
vqa_data = _module
vqa_model = _module
utils = _module

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


import copy


import logging


import math


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.nn import SmoothL1Loss


import collections


import random


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import DataLoader


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if (cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <=
        96 or cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]',
        '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if (cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or 
            cp >= 131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or
            cp >= 177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or
            cp >= 63744 and cp <= 64255 or cp >= 194560 and cp <= 195103):
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 65533 or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


PRETRAINED_VOCAB_ARCHIVE_MAP = {'bert-base-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
    , 'bert-large-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt'
    , 'bert-base-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt'
    , 'bert-large-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt'
    , 'bert-base-multilingual-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt'
    , 'bert-base-multilingual-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt'
    , 'bert-base-chinese':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt'
    }


PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {'bert-base-uncased': 512,
    'bert-large-uncased': 512, 'bert-base-cased': 512, 'bert-large-cased': 
    512, 'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512, 'bert-base-chinese': 512}


VOCAB_NAME = 'vocab.txt'


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


logger = logging.getLogger(__name__)


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise EnvironmentError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):

    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):

    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LXRTXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visual_attention = BertCrossattLayer(config)
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input,
        visn_attention_mask):
        lang_att_output = self.visual_attention(lang_input, visn_input,
            ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input,
            ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input,
        visn_attention_mask):
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats,
        visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(lang_att_output,
            lang_attention_mask, visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output,
            lang_attention_mask, visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output,
            visn_att_output)
        return lang_output, visn_output


class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']

    def __init__(self, l_layers=12, x_layers=5, r_layers=0):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = 2048
        self.visual_pos_dim = 4
        self.obj_id_num = 1600
        self.attr_id_num = 400
        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {'obj': (self.obj_id_num, 'ce', (-1,), 1 /
            0.15), 'attr': (self.attr_id_num, 'ce', (-1,), 1 / 0.15),
            'feat': (2048, 'l2', (-1, 2048), 1 / 0.15)}

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()


class VisualFeatEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2
        output = self.dropout(output)
        return output


class LXRTEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visn_fc = VisualFeatEncoder(config)
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        None
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.
            num_l_layers)])
        self.x_layers = nn.ModuleList([LXRTXLayer(config) for _ in range(
            self.num_x_layers)])
        self.r_layers = nn.ModuleList([BertLayer(config) for _ in range(
            self.num_r_layers)])

    def forward(self, lang_feats, lang_attention_mask, visn_feats,
        visn_attention_mask=None):
        visn_feats = self.visn_fc(visn_feats)
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats,
                lang_attention_mask, visn_feats, visn_attention_mask)
        return lang_feats, visn_feats


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.
            size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):

    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU
            (), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 
            2, num_answers))

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):

    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        visual_losses = visual_losses.split(',')
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses
        self.decoder_dict = nn.ModuleDict({key: nn.Linear(config.
            hidden_size, VISUAL_CONFIG.visual_loss_config[key][0]) for key in
            self.visual_losses})

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
            bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=
        3072, hidden_act='gelu', hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, max_position_embeddings=512,
        type_vocab_size=2, initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[
            0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
                ) as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
                )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


CONFIG_NAME = 'bert_config.json'


PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz'
    , 'bert-large-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz'
    , 'bert-base-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz'
    , 'bert-large-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz'
    , 'bert-base-multilingual-uncased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz'
    , 'bert-base-multilingual-cased':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz'
    , 'bert-base-chinese':
    'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'
    }


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print(
            'Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m'] for n in name):
            print('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        print('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.
                initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None,
        cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    None
                    archive_file = (
                        'https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz'
                        )
                    resolved_archive_file = cached_path(archive_file,
                        cache_dir=cache_dir)
                except EnvironmentError:
                    None
                    return None
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url."
                    .format(pretrained_model_name_or_path, ', '.join(
                    PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
        if resolved_archive_file == archive_file:
            logger.info('loading archive file {}'.format(archive_file))
        else:
            logger.info('loading archive file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not
                torch.cuda.is_available() else None)
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix
                [:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in
            state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'
                .format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):

    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=MAX_GQA_LENGTH)
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU
            (), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 
            2, num_answers))
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        return logit


class NLVR2Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=20)
        self.hid_dim = hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim * 2, hid_dim * 2),
            GeLU(), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(
            hid_dim * 2, 2))
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        :param feat: b, 2, o, f
        :param pos:  b, 2, o, 4
        :param sent: b, (string)
        :param leng: b, (numpy, int)
        :return:
        """
        sent = sum(zip(sent, sent), ())
        batch_size, img_num, obj_num, feat_size = feat.size()
        assert img_num == 2 and obj_num == 36 and feat_size == 2048
        feat = feat.view(batch_size * 2, obj_num, feat_size)
        pos = pos.view(batch_size * 2, obj_num, 4)
        x = self.lxrt_encoder(sent, (feat, pos))
        x = x.view(-1, self.hid_dim * 2)
        logit = self.logit_fc(x)
        return logit


MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):

    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=MAX_VQA_LENGTH)
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU
            (), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 
            2, num_answers))
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        return logit


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_airsplay_lxmert(_paritybench_base):
    pass
    def test_000(self):
        self._check(BertAttOutput(*[], **{'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BertEmbeddings(*[], **{'config': _mock_config(vocab_size=4, hidden_size=4, max_position_embeddings=4, type_vocab_size=4, hidden_dropout_prob=0.5)}), [torch.zeros([4, 4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(BertOutput(*[], **{'config': _mock_config(intermediate_size=4, hidden_size=4, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BertPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(BertVisualAnswerHead(*[], **{'config': _mock_config(hidden_size=4), 'num_answers': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(GeLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

