import sys
_module = sys.modules[__name__]
del sys
beam_search = _module
neural_combinatorial_rl = _module
plot_attention = _module
hyperparam_search = _module
plot_reward = _module
sorting_task = _module
trainer = _module
tsp_task = _module

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


import torch.autograd as autograd


from torch.autograd import Variable


import torch.nn.functional as F


import math


import numpy as np


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_hx = enc_init_hx
        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_cx = enc_init_cx
        return enc_init_hx, enc_init_cx


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()
        v = torch.FloatTensor(dim)
        if use_cuda:
            v = v
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)
        e = self.project_ref(ref)
        expanded_q = q.repeat(1, 1, e.size(2))
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)
            ).unsqueeze(1)
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class Beam(object):
    """Ordered beam of candidate outputs. Fixed length."""

    def __init__(self, size, steps, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = -1
        self.steps = steps
        self.current_step = 0
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.attn = []

    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]
        flat_beam_lk = beam_lk.view(-1)
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        self.current_step += 1
        if self.current_step == self.steps:
            self.done = True
        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]


class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, max_length,
        tanh_exploration, terminating_symbol, use_tanh, decode_type,
        n_glimpses=1, beam_size=0, use_cuda=True):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol
        self.decode_type = decode_type
        self.beam_size = beam_size
        self.use_cuda = use_cuda
        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=
            tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.
            use_cuda)
        self.sm = nn.Softmax()

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).byte()
            if self.use_cuda:
                mask = mask
        maskk = mask.clone()
        if prev_idxs is not None:
            maskk[[x for x in range(logits.size(0))], prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """

        def recurrence(x, hidden, logit_mask, prev_idxs, step):
            hx, cx = hidden
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)
            cy = forgetgate * cx + ingate * cellgate
            hy = outgate * F.tanh(cy)
            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits,
                    logit_mask, prev_idxs)
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)
            logits, logit_mask = self.apply_mask_to_logits(step, logits,
                logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)
        inps = []
        idxs = None
        mask = None
        if self.decode_type == 'stochastic':
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden,
                    mask, idxs, i)
                hidden = hx, cx
                decoder_input, idxs = self.decode_stochastic(probs,
                    embedded_inputs, selections)
                inps.append(decoder_input)
                outputs.append(probs)
                selections.append(idxs)
            return (outputs, selections), hidden
        elif self.decode_type == 'beam_search':
            decoder_input = Variable(decoder_input.data.repeat(self.
                beam_size, 1))
            context = Variable(context.data.repeat(1, self.beam_size, 1))
            hidden = Variable(hidden[0].data.repeat(self.beam_size, 1)
                ), Variable(hidden[1].data.repeat(self.beam_size, 1))
            beam = [Beam(self.beam_size, self.max_length, cuda=self.
                use_cuda) for k in range(batch_size)]
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden,
                    mask, idxs, i)
                hidden = hx, cx
                probs = probs.view(self.beam_size, batch_size, -1).transpose(
                    0, 1).contiguous()
                n_best = 1
                decoder_input, idxs, active = self.decode_beam(probs,
                    embedded_inputs, beam, batch_size, n_best, i)
                inps.append(decoder_input)
                if self.beam_size > 1:
                    outputs.append(probs[:, (0), :])
                else:
                    outputs.append(probs.squeeze(0))
                selections.append(idxs)
                if len(active) == 0:
                    break
                decoder_input = Variable(decoder_input.data.repeat(self.
                    beam_size, 1))
            return (outputs, selections), hidden

    def decode_stochastic(self, probs, embedded_inputs, selections):
        """
        Return the next input for the decoder by selecting the 
        input corresponding to the max output

        Args: 
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the 
            corresponding indicies
        """
        batch_size = probs.size(0)
        idxs = probs.multinomial().squeeze(1)
        for old_idxs in selections:
            if old_idxs.eq(idxs).data.any():
                None
                idxs = probs.multinomial().squeeze(1)
                break
        sels = embedded_inputs[(idxs.data), ([i for i in range(batch_size)]), :
            ]
        return sels, idxs

    def decode_beam(self, probs, embedded_inputs, beam, batch_size, n_best,
        step):
        active = []
        for b in range(batch_size):
            if beam[b].done:
                continue
            if not beam[b].advance(probs.data[b]):
                active += [b]
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
        all_idxs = Variable(torch.LongTensor([[x for x in hyp] for hyp in
            all_hyp]).squeeze())
        if all_idxs.dim() == 2:
            if all_idxs.size(1) > n_best:
                idxs = all_idxs[:, (-1)]
            else:
                idxs = all_idxs
        elif all_idxs.dim() == 3:
            idxs = all_idxs[:, (-1), :]
        elif all_idxs.size(0) > 1:
            idxs = all_idxs[-1]
        else:
            idxs = all_idxs
        if self.use_cuda:
            idxs = idxs
        if idxs.dim() > 1:
            x = embedded_inputs[(idxs.transpose(0, 1).contiguous().data), (
                [x for x in range(batch_size)]), :]
        else:
            x = embedded_inputs[(idxs.data), ([x for x in range(batch_size)
                ]), :]
        return x.view(idxs.size(0) * n_best, embedded_inputs.size(2)
            ), idxs, active


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""

    def __init__(self, embedding_dim, hidden_dim, max_decoding_len,
        terminating_symbol, n_glimpses, tanh_exploration, use_tanh,
        beam_size, use_cuda):
        super(PointerNetwork, self).__init__()
        self.encoder = Encoder(embedding_dim, hidden_dim, use_cuda)
        self.decoder = Decoder(embedding_dim, hidden_dim, max_length=
            max_decoding_len, tanh_exploration=tanh_exploration, use_tanh=
            use_tanh, terminating_symbol=terminating_symbol, decode_type=
            'stochastic', n_glimpses=n_glimpses, beam_size=beam_size,
            use_cuda=use_cuda)
        dec_in_0 = torch.FloatTensor(embedding_dim)
        if use_cuda:
            dec_in_0 = dec_in_0
        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-(1.0 / math.sqrt(embedding_dim)), 
            1.0 / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args: 
            inputs: [sourceL x batch_size x embedding_dim]
        """
        encoder_hx, encoder_cx = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1
            ).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1
            ).unsqueeze(0)
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx,
            encoder_cx))
        dec_init_state = enc_h_t[-1], enc_c_t[-1]
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1
            )
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
            inputs, dec_init_state, enc_h)
        return pointer_probs, input_idxs


class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self, embedding_dim, hidden_dim, n_process_block_iters,
        tanh_exploration, use_tanh, use_cuda):
        super(CriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters
        self.encoder = Encoder(embedding_dim, hidden_dim, use_cuda)
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=
            tanh_exploration, use_cuda=use_cuda)
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.
            ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        encoder_hx, encoder_cx = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1
            ).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1
            ).unsqueeze(0)
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx,
            encoder_cx))
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)
                ).squeeze(2)
        out = self.decoder(process_block_state)
        return out


class NeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific reward function
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim,
        max_decoding_len, terminating_symbol, n_glimpses,
        n_process_block_iters, tanh_exploration, use_tanh, beam_size,
        objective_fn, is_train, use_cuda):
        super(NeuralCombOptRL, self).__init__()
        self.objective_fn = objective_fn
        self.input_dim = input_dim
        self.is_train = is_train
        self.use_cuda = use_cuda
        self.actor_net = PointerNetwork(embedding_dim, hidden_dim,
            max_decoding_len, terminating_symbol, n_glimpses,
            tanh_exploration, use_tanh, beam_size, use_cuda)
        embedding_ = torch.FloatTensor(input_dim, embedding_dim)
        if self.use_cuda:
            embedding_ = embedding_
        self.embedding = nn.Parameter(embedding_)
        self.embedding.data.uniform_(-(1.0 / math.sqrt(embedding_dim)), 1.0 /
            math.sqrt(embedding_dim))

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, sourceL]
        """
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded_inputs = []
        ips = inputs.unsqueeze(1)
        for i in range(sourceL):
            embedded_inputs.append(torch.bmm(ips[:, :, :, (i)].float(),
                embedding).squeeze(1))
        embedded_inputs = torch.cat(embedded_inputs).view(sourceL,
            batch_size, embedding.size(2))
        probs_, action_idxs = self.actor_net(embedded_inputs)
        actions = []
        inputs_ = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs_[([x for x in range(batch_size)]), (
                action_id.data), :])
        if self.is_train:
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[[x for x in range(batch_size)], action_id
                    .data])
        else:
            probs = probs_
        R = self.objective_fn(actions, self.use_cuda)
        return R, probs, actions, action_idxs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pemami4911_neural_combinatorial_rl_pytorch(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(Decoder(*[], **{'embedding_dim': 4, 'hidden_dim': 4, 'max_length': 4, 'tanh_exploration': 4, 'terminating_symbol': 4, 'use_tanh': 4, 'decode_type': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(CriticNetwork(*[], **{'embedding_dim': 4, 'hidden_dim': 4, 'n_process_block_iters': 1, 'tanh_exploration': 4, 'use_tanh': 4, 'use_cuda': 4}), [torch.rand([4, 4, 4])], {})
