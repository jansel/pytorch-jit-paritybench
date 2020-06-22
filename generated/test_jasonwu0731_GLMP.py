import sys
_module = sys.modules[__name__]
del sys
GLMP = _module
modules = _module
myTest = _module
myTrain = _module
config = _module
masked_cross_entropy = _module
measures = _module
utils_Ent_babi = _module
utils_Ent_kvr = _module
utils_general = _module
utils_temp = _module

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


from torch.optim import lr_scheduler


from torch import optim


import torch.nn.functional as F


import random


import numpy as np


from torch.nn import functional


from torch.autograd import Variable


import torch.utils.data as data


import string


import re


import time


import math


import logging


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand
        )
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """
    if np.size(hypotheses) == 0:
        return np.float32(0.0)
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl'
            )
        os.chmod(multi_bleu_path, 493)
    except:
        print('Unable to fetch multi-bleu.perl script, using local.')
        metrics_dir = os.path.dirname(os.path.realpath(__file__))
        bin_dir = os.path.abspath(os.path.join(metrics_dir, '..', '..', 'bin'))
        multi_bleu_path = os.path.join(bin_dir, 'tools/multi-bleu.perl')
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write('\n'.join(hypotheses).encode('utf-8'))
    hypothesis_file.write(b'\n')
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write('\n'.join(references).encode('utf-8'))
    reference_file.write(b'\n')
    reference_file.flush()
    with open(hypothesis_file.name, 'r') as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ['-lc']
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred,
                stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode('utf-8')
            bleu_score = re.search('BLEU = (.+?),', bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print('multi-bleu.perl script returned non-zero exit code')
                print(error.output)
                bleu_score = np.float32(0.0)
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


_global_config['genSample'] = 4


_global_config['unk_mask'] = 4


_global_config['dataset'] = torch.rand([4, 4, 4, 4])


_global_config['addName'] = 4


_global_config['batch'] = 4


_global_config['teacher_forcing_ratio'] = 4


class GLMP(nn.Module):

    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr,
        n_layers, dropout):
        super(GLMP, self).__init__()
        self.name = 'GLMP'
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)
        if path:
            if USE_CUDA:
                None
                self.encoder = torch.load(str(path) + '/enc.th')
                self.extKnow = torch.load(str(path) + '/enc_kb.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                None
                self.encoder = torch.load(str(path) + '/enc.th', lambda
                    storage, loc: storage)
                self.extKnow = torch.load(str(path) + '/enc_kb.th', lambda
                    storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda
                    storage, loc: storage)
        else:
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.extKnow = ExternalKnowledge(lang.n_words, hidden_size,
                n_layers, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang,
                hidden_size, self.decoder_hop, dropout)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.
            decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=
            0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()
        if USE_CUDA:
            self.encoder
            self.extKnow
            self.decoder

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LE:{:.2f},LG:{:.2f},LP:{:.2f}'.format(print_loss_avg,
            print_loss_g, print_loss_v, print_loss_l)

    def save_model(self, dec_type):
        name_data = 'KVR/' if self.task == '' else 'BABI/'
        layer_info = str(self.n_layers)
        directory = 'save/GLMP-' + args['addName'] + name_data + str(self.task
            ) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(args['batch']
            ) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(self
            .lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        (self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l
            ) = 0, 1, 0, 0, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x)
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset:
            self.reset()
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        (all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _,
            global_pointer) = (self.encode_and_decode(data,
            max_target_length, use_teacher_forcing, False))
        loss_g = self.criterion_bce(global_pointer, data['selector_index'])
        loss_v = masked_cross_entropy(all_decoder_outputs_vocab.transpose(0,
            1).contiguous(), data['sketch_response'].contiguous(), data[
            'response_lengths'])
        loss_l = masked_cross_entropy(all_decoder_outputs_ptr.transpose(0, 
            1).contiguous(), data['ptr_index'].contiguous(), data[
            'response_lengths'])
        loss = loss_g + loss_v + loss_l
        loss.backward()
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def encode_and_decode(self, data, max_target_length,
        use_teacher_forcing, get_decoded_words):
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],
                story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, (0)] = rand_mask[:, :, (0)] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'
                    ][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end - start, (bi), :] = rand_mask[(bi),
                    start:end, :]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story = data['context_arr'], data['conv_arr']
        dh_outputs, dh_hidden = self.encoder(conv_story, data[
            'conv_arr_lengths'])
        global_pointer, kb_readout = self.extKnow.load_memory(story, data[
            'kb_arr_lengths'], data['conv_arr_lengths'], dh_hidden, dh_outputs)
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            self.copy_list.append(elm_temp)
        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = (self.
            decoder.forward(self.extKnow, story.size(), data[
            'context_arr_lengths'], self.copy_list, encoded_hidden, data[
            'sketch_response'], max_target_length, batch_size,
            use_teacher_forcing, get_decoded_words, global_pointer))
        return (outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse,
            global_pointer)

    def evaluate(self, dev, matric_best, early_stop=None):
        None
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)
        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0
        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ',
                            '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(
                                ' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        for j, data_dev in pbar:
            _, _, decoded_fine, decoded_coarse, global_pointer = (self.
                encode_and_decode(data_dev, self.max_resp_len, False, True))
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                if args['dataset'] == 'kvr':
                    single_f1, count = self.compute_prf(data_dev[
                        'ent_index'][bi], pred_sent.split(),
                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev[
                        'ent_idx_cal'][bi], pred_sent.split(),
                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev[
                        'ent_idx_nav'][bi], pred_sent.split(),
                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev[
                        'ent_idx_wet'][bi], pred_sent.split(),
                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                else:
                    current_id = data_dev['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)
                total += 1
                if gold_sent == pred_sent:
                    acc += 1
                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent,
                        pred_sent_coarse, gold_sent)
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref),
            lowercase=True)
        acc_score = acc / float(total)
        None
        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            None
            None
            None
            None
            None
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            None
        if early_stop == 'BLEU':
            if bleu_score >= matric_best:
                self.save_model('BLEU-' + str(bleu_score))
                None
            return bleu_score
        elif early_stop == 'ENTF1':
            if F1_score >= matric_best:
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                None
            return F1_score
        else:
            if acc_score >= matric_best:
                self.save_model('ACC-{:.4f}'.format(acc_score))
                None
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if TP + FP != 0 else 0
            recall = TP / float(TP + FN) if TP + FN != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall
                ) if precision + recall != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse,
        gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data[
            'conv_arr_lengths'][batch_idx] - 1
        None
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if
                w != 'PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                None
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][
            batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                None
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        None
        None
        None
        None


PAD_token = 1


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class ContextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=
            PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=
            dropout, bidirectional=True)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.
            size(0), -1).long())
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                batch_first=False)
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0, 1), hidden


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


_global_config['ablationH'] = 4


_global_config['ablationG'] = 4


class ExternalKnowledge(nn.Module):

    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module('C_{}'.format(hop), C)
        self.C = AttrProxy(self, 'C_')
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[(bi), start:end, :] = full_memory[(bi), start:end, :
                ] + hiddens[(bi), :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs):
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))
            embed_A = torch.sum(embed_A, 2).squeeze(2)
            if not args['ablationH']:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len,
                    dh_outputs)
            embed_A = self.dropout_layer(embed_A)
            if len(list(u[-1].size())) == 1:
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)
            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0],
                -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args['ablationH']:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len,
                    dh_outputs)
            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if not args['ablationG']:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if len(list(u[-1].size())) == 1:
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[hop + 1]
            if not args['ablationG']:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


SOS_token = 3


_global_config['record'] = 4


class LocalMemoryDecoder(nn.Module):

    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2 * embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, extKnow, story_size, story_lengths, copy_list,
        encode_hidden, target_batches, max_target_length, batch_size,
        use_teacher_forcing, get_decoded_words, global_pointer):
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length,
            batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length,
            batch_size, story_size[1]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input))
            if len(embed_q.size()) == 1:
                embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]
            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)
            prob_soft, prob_logits = extKnow(query_vector, global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits
            if use_teacher_forcing:
                decoder_input = target_batches[:, (t)]
            else:
                decoder_input = topvi.squeeze()
            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []
                for bi in range(batch_size):
                    token = topvi[bi].item()
                    temp_c.append(self.lang.index2word[token])
                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, (i)][bi] < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[:, (i)][bi].item()]
                                break
                        temp_f.append(cw)
                        if args['record']:
                            memory_mask_for_step[bi, toppi[:, (i)][bi].item()
                                ] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])
                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)
        return (all_decoder_outputs_vocab, all_decoder_outputs_ptr,
            decoded_fine, decoded_coarse)

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        return scores_


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jasonwu0731_GLMP(_paritybench_base):
    pass
