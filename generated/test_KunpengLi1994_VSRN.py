import sys
_module = sys.modules[__name__]
del sys
Rs_GCN = _module
GCN_lib = _module
master = _module
pyciderevalcap = _module
cider = _module
cider_scorer = _module
ciderD = _module
ciderD_scorer = _module
eval = _module
tokenizer = _module
ptbtokenizer = _module
pycocoevalcap = _module
bleu = _module
bleu_scorer = _module
meteor = _module
rouge = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
setup = _module
data = _module
evaluate_models = _module
evaluation = _module
evaluation_models = _module
misc = _module
rewards = _module
utils = _module
model = _module
Attention = _module
DecoderRNN = _module
EncoderRNN = _module
S2VTAttModel = _module
S2VTModel = _module
models = _module
opts = _module
train = _module
vocab = _module

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


from torch import nn


from torch.nn import functional as F


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.init


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.backends.cudnn as cudnn


from torch.nn.utils.clip_grad import clip_grad_norm


import numpy as np


from collections import OrderedDict


import torch.nn.functional as F


import torch.optim as optim


import random


class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels,
                out_channels=self.in_channels, kernel_size=1, stride=1,
                padding=0), bn(self.in_channels))
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=
                self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)
        self.theta = None
        self.phi = None
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, v):
        """
        :param v: (B, D, N)
        :return:
        """
        batch_size = v.size(0)
        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)
        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N
        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v
        return v_star


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1
            ).contiguous().view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
        use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.cnn = self.get_cnn(cnn_type, True)
        for param in self.cnn.parameters():
            param.requires_grad = finetune
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].
                in_features, embed_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.
                children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            None
            model = models.__dict__[arch](pretrained=True)
        else:
            None
            model = models.__dict__[arch]()
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model
        else:
            model = nn.DataParallel(model)
        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']
        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)
        features = l2norm(features)
        features = self.fc(features)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, img_dim, embed_size, data_name, use_abs=False,
        no_imgnorm=False):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=
            embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=
            embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=
            embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=
            embed_size)
        if self.data_name == 'f30k_precomp':
            self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd)
        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)
        features = hidden_state[0]
        if self.data_name == 'f30k_precomp':
            features = self.bn(features)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
        use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1)
        out = torch.gather(padded[0], 1, I).squeeze(1)
        out = l2norm(out)
        if self.use_abs:
            out = torch.abs(out)
        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)
        ) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 
            self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, n_layers=
        1, rnn_cell='gru', bidirectional=False, input_dropout_p=0.1,
        rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden + dim_word, self.
            dim_hidden, n_layers, batch_first=True, dropout=rnn_dropout_p)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        self._init_weights()

    def forward(self, encoder_outputs, encoder_hidden, targets=None, mode=
        'train', opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        seq_logprobs = []
        seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, (i), :]
                context = self.attention(decoder_hidden.squeeze(0),
                    encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input,
                    decoder_hidden)
                logprobs = F.log_softmax(self.out(decoder_output.squeeze(1)
                    ), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))
            seq_logprobs = torch.cat(seq_logprobs, 1)
        elif mode == 'inference':
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)
            for t in range(self.max_length - 1):
                context = self.attention(decoder_hidden.squeeze(0),
                    encoder_outputs)
                if t == 0:
                    it = torch.LongTensor([self.sos_id] * batch_size)
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()
                seq_preds.append(it.view(-1, 1))
                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input,
                    decoder_hidden)
                logprobs = F.log_softmax(self.out(decoder_output.squeeze(1)
                    ), dim=1)
            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)
        return seq_logprobs, seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in
                encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class EncoderRNN(nn.Module):

    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2,
        rnn_dropout_p=0.5, n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers,
            batch_first=True, bidirectional=bidirectional, dropout=self.
            rnn_dropout_p)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden


class S2VTAttModel(nn.Module):

    def __init__(self, encoder, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None, mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden,
            target_variable, mode, opt)
        return seq_prob, seq_preds


class S2VTModel(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=
        2048, sos_id=1, eos_id=0, n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2
        ):
        super(S2VTModel, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden,
            n_layers, batch_first=True, dropout=rnn_dropout_p)
        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def forward(self, vid_feats, target_variable=None, mode='train', opt={}):
        batch_size, n_frames, _ = vid_feats.shape
        padding_words = Variable(vid_feats.data.new(batch_size, n_frames,
            self.dim_word)).zero_()
        padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.
            dim_vid)).zero_()
        state1 = None
        state2 = None
        output1, state1 = self.rnn1(vid_feats, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)
        seq_probs = []
        seq_preds = []
        if mode == 'train':
            for i in range(self.max_length - 1):
                current_words = self.embedding(target_variable[:, (i)])
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2
                    )
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
        else:
            current_words = self.embedding(Variable(torch.LongTensor([self.
                sos_id] * batch_size)))
            for i in range(self.max_length - 1):
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2
                    )
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KunpengLi1994_VSRN(_paritybench_base):
    pass
    def test_000(self):
        self._check(Attention(*[], **{'dim': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ContrastiveLoss(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_002(self):
        self._check(EncoderImagePrecomp(*[], **{'img_dim': 4, 'embed_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(EncoderImagePrecompAttn(*[], **{'img_dim': 4, 'embed_size': 4, 'data_name': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(EncoderRNN(*[], **{'dim_vid': 4, 'dim_hidden': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(RewardCriterion(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(Rs_GCN(*[], **{'in_channels': 4, 'inter_channels': 4}), [torch.rand([4, 4, 64])], {})

