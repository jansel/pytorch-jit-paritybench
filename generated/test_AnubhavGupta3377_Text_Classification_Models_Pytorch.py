import sys
_module = sys.modules[__name__]
del sys
config = _module
model = _module
train = _module
utils = _module
model = _module
model = _module
model = _module
model = _module
model = _module
model = _module
attention = _module
encoder = _module
feed_forward = _module
model = _module
sublayer = _module
train_utils = _module
model = _module
model = _module
reformat_data = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


import numpy as np


from torch.autograd import Variable


from torch.nn import functional as F


from torch import Tensor


import math


import torch.nn.functional as F


import torch.nn as nn


from copy import deepcopy


import copy


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score


class CharCNN(nn.Module):

    def __init__(self, config, vocab_size, embeddings):
        super(CharCNN, self).__init__()
        self.config = config
        embed_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)
        conv1 = nn.Sequential(nn.Conv1d(in_channels=embed_size, out_channels=self.config.num_channels, kernel_size=7), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv2 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=7), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv3 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv5 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv_output_size = self.config.num_channels * ((self.config.seq_len - 96) // 27)
        linear1 = nn.Sequential(nn.Linear(conv_output_size, self.config.linear_size), nn.ReLU(), nn.Dropout(self.config.dropout_keep))
        linear2 = nn.Sequential(nn.Linear(self.config.linear_size, self.config.linear_size), nn.ReLU(), nn.Dropout(self.config.dropout_keep))
        linear3 = nn.Sequential(nn.Linear(self.config.linear_size, self.config.output_size), nn.Softmax())
        self.convolutional_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 2, 0)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch > 0 and epoch % 3 == 0:
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = batch.label.type(torch.LongTensor)
            else:
                x = batch.text
                y = batch.label.type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class CharCNN(nn.Module):

    def __init__(self, config):
        super(CharCNN, self).__init__()
        self.config = config
        conv1 = nn.Sequential(nn.Conv1d(in_channels=self.config.vocab_size, out_channels=self.config.num_channels, kernel_size=7), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv2 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=7), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv3 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv5 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3), nn.ReLU(), nn.MaxPool1d(kernel_size=3))
        conv_output_size = self.config.num_channels * ((self.config.max_len - 96) // 27)
        linear1 = nn.Sequential(nn.Linear(conv_output_size, self.config.linear_size), nn.ReLU(), nn.Dropout(self.config.dropout_keep))
        linear2 = nn.Sequential(nn.Linear(self.config.linear_size, self.config.linear_size), nn.ReLU(), nn.Dropout(self.config.dropout_keep))
        linear3 = nn.Sequential(nn.Linear(self.config.linear_size, self.config.output_size), nn.Softmax())
        self.convolutional_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, embedded_sent):
        embedded_sent = embedded_sent.transpose(1, 2)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch > 0 and epoch % 3 == 0:
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            _, n_true_label = batch
            if torch.is_available():
                batch = [Variable(record) for record in batch]
            else:
                batch = [Variable(record) for record in batch]
            x, y = batch
            self.optimizer.zero_grad()
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                self.eval()
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class RCNN(nn.Module):

    def __init__(self, config, vocab_size, word_embeddings):
        super(RCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(input_size=self.config.embed_size, hidden_size=self.config.hidden_size, num_layers=self.config.hidden_layers, dropout=self.config.dropout_keep, bidirectional=True)
        self.dropout = nn.Dropout(self.config.dropout_keep)
        self.W = nn.Linear(self.config.embed_size + 2 * self.config.hidden_size, self.config.hidden_size_linear)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.config.hidden_size_linear, self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)
        input_features = torch.cat([lstm_out, embedded_sent], 2).permute(1, 0, 2)
        linear_output = self.tanh(self.W(input_features))
        linear_output = linear_output.permute(0, 2, 1)
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class Seq2SeqAttention(nn.Module):

    def __init__(self, config, vocab_size, word_embeddings):
        super(Seq2SeqAttention, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(input_size=self.config.embed_size, hidden_size=self.config.hidden_size, num_layers=self.config.hidden_layers, bidirectional=self.config.bidirectional)
        self.dropout = nn.Dropout(self.config.dropout_keep)
        self.fc = nn.Linear(self.config.hidden_size * (1 + self.config.bidirectional) * 2, self.config.output_size)
        self.softmax = nn.Softmax()

    def apply_attention(self, rnn_output, final_hidden_state):
        """
        Apply Attention on RNN output
        
        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
            
        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        """
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, x):
        embedded_sent = self.embeddings(x)
        lstm_output, (h_n, c_n) = self.lstm(embedded_sent)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.config.hidden_layers, self.config.bidirectional + 1, batch_size, self.config.hidden_size)[(-1), :, :, :]
        final_hidden_state = torch.cat([h_n_final_layer[(i), :, :] for i in range(h_n_final_layer.shape[0])], dim=1)
        attention_out = self.apply_attention(lstm_output.permute(1, 0, 2), final_hidden_state)
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class TextCNN(nn.Module):

    def __init__(self, config, vocab_size, word_embeddings):
        super(TextCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels, kernel_size=self.config.kernel_size[0]), nn.ReLU(), nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels, kernel_size=self.config.kernel_size[1]), nn.ReLU(), nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels, kernel_size=self.config.kernel_size[2]), nn.ReLU(), nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1))
        self.dropout = nn.Dropout(self.config.dropout_keep)
        self.fc = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 2, 0)
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


def data_iterator(train_x, train_y, batch_size=256):
    """
    Generate batches of training data for training (for single epoch)
    Inputs:
        train_df (pd.DataFrame) : complete training data
        batch_size (int) : Size of each batch
    Returns:
        text_arr (np.matrix) : Matrix of shape (batch_size,embed_size)
        lebel_arr (np.array) : Labels of this batch. Array of shape (batch_size,)
    """
    n_batches = math.ceil(len(train_x) / batch_size)
    for idx in range(n_batches):
        x = train_x[idx * batch_size:(idx + 1) * batch_size]
        y = train_y[idx * batch_size:(idx + 1) * batch_size]
        yield x, y


class CNNText(nn.Module):

    def __init__(self, config):
        super(CNNText, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels, kernel_size=(self.config.kernel_size[0], self.config.embed_size), stride=1, padding=0)
        self.activation1 = nn.ReLU()
        self.max_out1 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1)
        self.conv2 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels, kernel_size=(self.config.kernel_size[1], self.config.embed_size), stride=1, padding=0)
        self.activation2 = nn.ReLU()
        self.max_out2 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1)
        self.conv3 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels, kernel_size=(self.config.kernel_size[2], self.config.embed_size), stride=1, padding=0)
        self.activation3 = nn.ReLU()
        self.max_out3 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1)
        self.dropout = nn.Dropout(self.config.dropout_keep)
        self.fc = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_out1 = self.conv1(x).squeeze(3)
        activation_out1 = self.activation1(conv_out1)
        max_out1 = self.max_out1(activation_out1).squeeze(2)
        conv_out2 = self.conv2(x).squeeze(3)
        activation_out2 = self.activation2(conv_out2)
        max_out2 = self.max_out2(activation_out2).squeeze(2)
        conv_out3 = self.conv3(x).squeeze(3)
        activation_out3 = self.activation3(conv_out3)
        max_out3 = self.max_out3(activation_out3).squeeze(2)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def run_epoch(self, train_data, val_data):
        train_x, train_y = train_data[0], train_data[1]
        val_x, val_y = val_data[0], val_data[1]
        iterator = data_iterator(train_x, train_y, self.config.batch_size)
        train_losses = []
        val_accuracies = []
        losses = []
        for i, (x, y) in enumerate(iterator):
            self.optimizer.zero_grad()
            x = Tensor(x)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, torch.LongTensor(y - 1))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if (i + 1) % 50 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                self.eval()
                all_preds = []
                val_iterator = data_iterator(val_x, val_y, self.config.batch_size)
                for j, (x, y) in enumerate(val_iterator):
                    x = Variable(Tensor(x))
                    y_pred = self.__call__(x)
                    predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
                    all_preds.extend(predicted.numpy())
                score = accuracy_score(val_y, np.array(all_preds).flatten())
                val_accuracies.append(score)
                None
                self.train()
        return train_losses, val_accuracies


class TextRNN(nn.Module):

    def __init__(self, config, vocab_size, word_embeddings):
        super(TextRNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(input_size=self.config.embed_size, hidden_size=self.config.hidden_size, num_layers=self.config.hidden_layers, dropout=self.config.dropout_keep, bidirectional=self.config.bidirectional)
        self.dropout = nn.Dropout(self.config.dropout_keep)
        self.fc = nn.Linear(self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional), self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)
        final_feature_map = self.dropout(h_n)
        final_feature_map = torch.cat([final_feature_map[(i), :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


def attention(query, key, value, mask=None, dropout=None):
    """Implementation of Scaled dot product attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Multi-head attention"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Encoder(nn.Module):
    """
    Transformer Encoder
    
    It is a stack of N layers.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    An encoder layer
    
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        """Transformer Encoder"""
        x = self.sublayer_output[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Implements FFN equation."""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Transformer(nn.Module):

    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab), deepcopy(position))
        self.fc = nn.Linear(self.config.d_model, self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1, 0))
        encoded_sents = self.encoder(embedded_sents)
        final_feature_map = encoded_sents[:, (-1), :]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class LayerNorm(nn.Module):
    """Construct a layer normalization module."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    """
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class fastText(nn.Module):

    def __init__(self, config, vocab_size, word_embeddings):
        super(fastText, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 0, 2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)
        return self.softmax(z)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        None
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        if epoch == int(self.config.max_epochs / 3) or epoch == int(2 * self.config.max_epochs / 3):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.is_available():
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if i % 100 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                val_accuracy = evaluate_model(self, val_iterator)
                None
                self.train()
        return train_losses, val_accuracies


class fastText(nn.Module):

    def __init__(self, config):
        super(fastText, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.fc1(x)
        z = self.fc2(h)
        return self.softmax(z)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def run_epoch(self, train_data, val_data):
        train_x, train_y = train_data[0], train_data[1]
        val_x, val_y = val_data[0], val_data[1]
        iterator = data_iterator(train_x, train_y, self.config.batch_size)
        train_losses = []
        val_accuracies = []
        losses = []
        for i, (x, y) in enumerate(iterator):
            self.optimizer.zero_grad()
            x = Tensor(x)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, torch.LongTensor(y - 1))
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            if (i + 1) % 50 == 0:
                None
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                None
                losses = []
                self.eval()
                all_preds = []
                val_iterator = data_iterator(val_x, val_y, self.config.batch_size)
                for x, y in val_iterator:
                    x = Variable(Tensor(x))
                    y_pred = self.__call__(x)
                    predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
                    all_preds.extend(predicted.numpy())
                score = accuracy_score(val_y, np.array(all_preds).flatten())
                val_accuracies.append(score)
                None
                self.train()
        return train_losses, val_accuracies


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Embeddings,
     lambda: ([], {'d_model': 4, 'vocab': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SublayerOutput,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (fastText,
     lambda: ([], {'config': _mock_config(embed_size=4, hidden_size=4, output_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_AnubhavGupta3377_Text_Classification_Models_Pytorch(_paritybench_base):
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

