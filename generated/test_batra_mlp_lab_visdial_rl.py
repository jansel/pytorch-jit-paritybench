import sys
_module = sys.modules[__name__]
del sys
prepro = _module
dataloader = _module
eval_utils = _module
dialog_generate = _module
rank_answerer = _module
rank_questioner = _module
evaluate = _module
options = _module
train = _module
utils = _module
utilities = _module
visualize = _module
visdial = _module
metrics = _module
models = _module
agent = _module
answerer = _module
decoders = _module
gen = _module
encoders = _module
hre = _module
questioner = _module

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


import numpy as np


import torch


from sklearn.preprocessing import normalize


from torch.utils.data import Dataset


from torch.autograd import Variable


from torch.utils.data import DataLoader


from sklearn.metrics.pairwise import pairwise_distances


import torch.nn.functional as F


import random


from time import gmtime


from time import strftime


import torch.nn as nn


import torch.optim as optim


import math


from torch.nn.utils.rnn import pack_padded_sequence


from torch.distributions import Categorical


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()


class Answerer(Agent):

    def __init__(self, encoderParam, decoderParam, verbose=1):
        """
            A-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (answer).
        """
        super(Answerer, self).__init__()
        self.encType = encoderParam['type']
        self.decType = decoderParam['type']
        if verbose:
            None
            None
        if 'hre' in self.encType:
            self.encoder = hre_enc.Encoder(**encoderParam)
        else:
            raise Exception('Unknown encoder {}'.format(self.encType))
        if 'gen' == self.decType:
            self.decoder = gen_dec.Decoder(**decoderParam)
        else:
            raise Exception('Unkown decoder {}'.format(self.decType))
        self.decoder.wordEmbed = self.encoder.wordEmbed
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.reset()

    def reset(self):
        """Delete dialog history."""
        self.caption = None
        self.answers = []
        self.encoder.reset()

    def observe(self, round, ans=None, caption=None, **kwargs):
        """
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        """
        if caption is not None:
            assert round == -1, 'Round number should be -1 when observing caption, got %d instead'
            self.caption = caption
        if ans is not None:
            assert round == len(self.answers), 'Round number does not match number of answers observed'
            self.answers.append(ans)
        self.encoder.observe(round, ans=ans, caption=caption, **kwargs)

    def forward(self):
        """
        Forward pass the last observed answer to compute its log
        likelihood under the current decoder RNN state.
        """
        encStates = self.encoder()
        if len(self.answers) > 0:
            decIn = self.answers[-1]
        elif self.caption is not None:
            decIn = self.caption
        else:
            raise Exception('Must provide an input sequence')
        logProbs = self.decoder(encStates, inputSeq=decIn)
        return logProbs

    def forwardDecode(self, inference='sample', beamSize=1, maxSeqLen=20):
        """
        Decode a sequence (answer) using either sampling or greedy inference.
        An answer is decoded given the current state (dialog history). This
        can be called at every round after a question is observed.

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        """
        encStates = self.encoder()
        answers, ansLens = self.decoder.forwardDecode(encStates, maxSeqLen=maxSeqLen, inference=inference, beamSize=beamSize)
        return answers, ansLens

    def evalOptions(self, options, optionLens, scoringFunction):
        """
        Given the current state (question and conversation history), evaluate
        a set of candidate answers to the question.

        Output:
            Log probabilities of candidate options.
        """
        states = self.encoder()
        return self.decoder.evalOptions(states, options, optionLens, scoringFunction)

    def reinforce(self, reward):
        return self.decoder.reinforce(reward)


class Decoder(nn.Module):

    def __init__(self, vocabSize, embedSize, rnnHiddenSize, numLayers, startToken, endToken, dropout=0, **kwargs):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.startToken = startToken
        self.endToken = endToken
        self.dropout = dropout
        self.rnn = nn.LSTM(self.embedSize, self.rnnHiddenSize, self.numLayers, batch_first=True, dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, encStates, inputSeq):
        """
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.

        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities

        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at evert time step

        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        """
        if inputSeq is not None:
            inputSeq = self.wordEmbed(inputSeq)
            outputs, _ = self.rnn(inputSeq, encStates)
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputSize = outputs.size()
            flatOutputs = outputs.view(-1, outputSize[2])
            flatScores = self.outNet(flatOutputs)
            flatLogProbs = self.logSoftmax(flatScores)
            logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)
        return logProbs

    def forwardDecode(self, encStates, maxSeqLen=20, inference='sample', beamSize=1):
        """
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width

        Notes:
            * This function is not called during SL pre-training
            * Greedy inference is used for evaluation
            * Sampling is used in RL fine-tuning
        """
        if inference == 'greedy' and beamSize > 1:
            return self.beamSearchDecoder(encStates, beamSize, maxSeqLen)
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch
        self.samples = []
        maxLen = maxSeqLen + 1
        batchSize = encStates[0].size(1)
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, (0)] = self.startToken
        seq = Variable(seq, requires_grad=False)
        hid = encStates
        sampleLens = th.LongTensor(batchSize).fill_(0)
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.ByteTensor(seq.size()).fill_(0)
        self.saved_log_probs = []
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            output, hid = self.rnn(emb, hid)
            scores = self.outNet(output.squeeze(1))
            logProb = self.logSoftmax(scores)
            if t > 0:
                logProb = torch.cat([logProb[:, 1:-2], logProb[:, -1:]], 1)
            elif t == 0:
                logProb = logProb[:, 1:-2]
            END_TOKEN_IDX = self.endToken - 1
            probs = torch.exp(logProb)
            if inference == 'sample':
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                self.saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)
            elif inference == 'greedy':
                _, sample = torch.max(probs, dim=1, keepdim=True)
            else:
                raise ValueError("Invalid inference type: '{}'".format(inference))
            sample = sample + 1
            self.samples.append(sample)
            seq.data[:, (t + 1)] = sample.data
            mask[:, (t)] = sample.data.eq(END_TOKEN_IDX)
            sample.data.masked_fill_(mask[:, (t)].unsqueeze(1), self.endToken)
        mask[:, (maxLen - 1)].fill_(1)
        for t in range(maxLen):
            unitColumn.masked_fill_(mask[:, (t)], 0)
            mask[:, (t)] = unitColumn
            sampleLens = sampleLens + unitColumn
        self.mask = Variable(mask, requires_grad=False)
        sampleLens = sampleLens + 1
        sampleLens = Variable(sampleLens, requires_grad=False)
        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)
        gen_samples = [startColumn] + self.samples
        samples = torch.cat(gen_samples, 1)
        return samples, sampleLens

    def evalOptions(self, encStates, options, optionLens, scoringFunction):
        """
        Forward pass a set of candidate options to get log probabilities

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            options   : (batchSize, numOptions, maxSequenceLength) sized
                        tensor with <START> and <END> tokens

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.

        Output:
            A (batchSize, numOptions) tensor containing the score
            of each option sentence given by the generator
        """
        batchSize, numOptions, maxLen = options.size()
        optionsFlat = options.contiguous().view(-1, maxLen)
        encStates = [x.unsqueeze(2).repeat(1, 1, numOptions, 1).view(self.numLayers, -1, self.rnnHiddenSize) for x in encStates]
        logProbs = self.forward(encStates, inputSeq=optionsFlat)
        scores = scoringFunction(logProbs, optionsFlat, returnScores=True)
        return scores.view(batchSize, numOptions)

    def reinforce(self, reward):
        """
        Compute loss using REINFORCE on log probabilities of tokens
        sampled from decoder RNN, scaled by input 'reward'.

        Note that an earlier call to forwardDecode must have been
        made in order to have samples for which REINFORCE can be
        applied. These samples are stored in 'self.saved_log_probs'.
        """
        loss = 0
        if len(self.saved_log_probs) == 0:
            raise RuntimeError('Reinforce called without sampling in Decoder')
        for t, log_prob in enumerate(self.saved_log_probs):
            loss += -1 * log_prob * (reward * self.mask[:, (t)].float())
        return loss

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen):
        """
        Beam search for sequence generation

        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        """
        assert self.training == False
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch
        LENGTH_NORM = True
        maxLen = maxSeqLen + 1
        batchSize = initStates[0].size(1)
        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)
        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange)
        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = initStates
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(self.endToken)
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)
        aliveVector = beamTokensTable[:, :, (0)].eq(self.endToken).unsqueeze(2)
        for t in range(maxLen - 1):
            if t == 0:
                emb = self.wordEmbed(startTokenArray)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores)
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                beamTokensTable[:, :, (0)] = topIdx.transpose(0, 1).data
                logProbSums = topLogProbs
                hiddenStates = [x.unsqueeze(2).repeat(1, 1, beamSize, 1) for x in hiddenStates]
                hiddenStates = [x.view(self.numLayers, -1, self.rnnHiddenSize) for x in hiddenStates]
            else:
                emb = self.wordEmbed(beamTokensTable[:, :, (t - 1)])
                output, hiddenStates = self.rnn(emb.view(-1, 1, self.embedSize), hiddenStates)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize, self.vocabSize)
                if LENGTH_NORM:
                    logProbs = logProbsCurrent * (aliveVector.float() / (t + 1))
                    coeff_ = aliveVector.eq(0).float() + aliveVector.float() * t / (t + 1)
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    logProbs = logProbsCurrent * aliveVector.float()
                    logProbs += logProbSums.unsqueeze(2)
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :, (0)] = 0
                minus_infinity_ = torch.min(logProbs).data[0]
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)
                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batchSize, beamSize, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocabSize).view(batchSize, -1)
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                logProbSums = topLogProbs
                beamTokensTable[:, :, (t)] = tokensArray.gather(1, topIdx)
                backIndices[:, :, (t)] = backIndexArray.gather(1, topIdx)
                hiddenCurrent, cellCurrent = hiddenStates
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(num_layers, batchSize, beamSize, rnnHiddenSize)
                backIndexVector = backIndices[:, :, (t)].unsqueeze(0).unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = hiddenCurrent, cellCurrent
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data
        RECOVER_TOP_BEAM_ONLY = True
        tokenIdx = finalLen
        backID = backIndices[:, :, (tokenIdx)]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, (tokenIdx)].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, (tokenIdx)].gather(1, backID)
            tokenIdx = tokenIdx - 1
        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)
        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, (0)]
            seqLens = seqLens[:, (0)]
        return Variable(tokens), Variable(seqLens)


class Encoder(nn.Module):

    def __init__(self, vocabSize, embedSize, rnnHiddenSize, numLayers, useIm, imgEmbedSize, imgFeatureSize, numRounds, isAnswerer, dropout=0, startToken=None, endToken=None, **kwargs):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, 'Less than 2 layers not supported!'
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken
        self.wordEmbed = nn.Embedding(self.vocabSize, self.embedSize, padding_idx=0)
        if self.useIm == 'early':
            quesInputSize = self.embedSize + self.imgEmbedSize
            dialogInputSize = 2 * self.rnnHiddenSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == 'late':
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize + self.imgEmbedSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.isAnswerer:
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        else:
            dialogInputSize = self.rnnHiddenSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(quesInputSize, self.rnnHiddenSize, self.numLayers, batch_first=True, dropout=0)
        self.factRNN = nn.LSTM(self.embedSize, self.rnnHiddenSize, self.numLayers, batch_first=True, dropout=0)
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

    def reset(self):
        self.batchSize = 0
        self.image = None
        self.imageEmbed = None
        self.captionTokens = None
        self.captionEmbed = None
        self.captionLens = None
        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []
        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []

    def _initHidden(self):
        """Initial dialog rnn state - initialize with zeros"""
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return Variable(h), Variable(c)

    def observe(self, round, image=None, caption=None, ques=None, ans=None, captionLens=None, quesLens=None, ansLens=None):
        """
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        """
        if image is not None:
            assert round == -1
            self.image = image
            self.imageEmbed = None
            self.batchSize = len(self.image)
        if caption is not None:
            assert round == -1
            assert captionLens is not None, 'Caption lengths required!'
            caption, captionLens = self.processSequence(caption, captionLens)
            self.captionTokens = caption
            self.captionLens = captionLens
            self.batchSize = len(self.captionTokens)
        if ques is not None:
            assert round == len(self.questionEmbeds)
            assert quesLens is not None, 'Questions lengths required!'
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, 'Answer lengths required!'
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)

    def processSequence(self, seq, seqLen):
        """ Strip <START> and <END> token from a left-aligned sequence"""
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        """
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        """
        if self.isAnswerer and self.imageEmbed is None:
            self.imageEmbed = self.imgNet(self.imgEmbedDropout(self.image))
        if self.captionEmbed is None:
            self.captionEmbed = self.wordEmbed(self.captionTokens)
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(self.wordEmbed(self.questionTokens[idx]))
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        """Embed facts i.e. caption and round 0 or question-answer pair otherwise"""
        if factIdx == 0:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(self.factRNN, seq, seqLens, returnStates=True)
        elif factIdx > 0:
            quesTokens, quesLens = self.questionTokens[factIdx - 1], self.questionLens[factIdx - 1]
            ansTokens, ansLens = self.answerTokens[factIdx - 1], self.answerLengths[factIdx - 1]
            qaTokens = utils.concatPaddedSequences(quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        """Embed questions"""
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
        if self.useIm == 'late':
            currIns.append(self.imageEmbed)
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)

    def forward(self):
        """
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        """
        self.embedInputDialog()
        if self.isAnswerer:
            round = len(self.questionEmbeds) - 1
        else:
            round = len(self.answerEmbeds)
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx)
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)
        dialogHidden = self.dialogHiddens[-1][0]
        """
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)
              Layer 1 : DialogRNN hidden state (dialogRNN)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN)
        """
        if self.isAnswerer:
            quesRNNstates = self.questionRNNStates[-1][1]
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            factRNNstates = self.factEmbeds[-1][1]
            C_link = factRNNstates[1]
            H_link = factRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        return H_link, C_link


class Questioner(Agent):

    def __init__(self, encoderParam, decoderParam, imgFeatureSize=0, verbose=1):
        """
            Q-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (question).
        """
        super(Questioner, self).__init__()
        self.encType = encoderParam['type']
        self.decType = decoderParam['type']
        self.dropout = encoderParam['dropout']
        self.rnnHiddenSize = encoderParam['rnnHiddenSize']
        self.imgFeatureSize = imgFeatureSize
        encoderParam = encoderParam.copy()
        encoderParam['isAnswerer'] = False
        if verbose:
            None
            None
        if 'hre' in self.encType:
            self.encoder = hre_enc.Encoder(**encoderParam)
        else:
            raise Exception('Unknown encoder {}'.format(self.encType))
        if 'gen' == self.decType:
            self.decoder = gen_dec.Decoder(**decoderParam)
        else:
            raise Exception('Unkown decoder {}'.format(self.decType))
        self.decoder.wordEmbed = self.encoder.wordEmbed
        if self.imgFeatureSize:
            self.featureNet = nn.Linear(self.rnnHiddenSize, self.imgFeatureSize)
            self.featureNetInputDropout = nn.Dropout(0.5)
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.reset()

    def reset(self):
        """Delete dialog history."""
        self.questions = []
        self.encoder.reset()

    def freezeFeatNet(self):
        nets = [self.featureNet]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = False

    def observe(self, round, ques=None, **kwargs):
        """
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        """
        assert 'image' not in kwargs, 'Q-Bot does not see image'
        if ques is not None:
            assert round == len(self.questions), 'Round number does not match number of questions observed'
            self.questions.append(ques)
        self.encoder.observe(round, ques=ques, **kwargs)

    def forward(self):
        """
        Forward pass the last observed question to compute its log
        likelihood under the current decoder RNN state.
        """
        encStates = self.encoder()
        if len(self.questions) == 0:
            raise Exception('Must provide question if not sampling one.')
        decIn = self.questions[-1]
        logProbs = self.decoder(encStates, inputSeq=decIn)
        return logProbs

    def forwardDecode(self, inference='sample', beamSize=1, maxSeqLen=20):
        """
        Decode a sequence (question) using either sampling or greedy inference.
        A question is decoded given current state (dialog history). This can
        be called at round 0 after the caption is observed, and at end of every
        round (after a response from A-Bot is observed).

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        """
        encStates = self.encoder()
        questions, quesLens = self.decoder.forwardDecode(encStates, maxSeqLen=maxSeqLen, inference=inference, beamSize=beamSize)
        return questions, quesLens

    def predictImage(self):
        """
        Predict/guess an fc7 vector given the current conversation history. This can
        be called at round 0 after the caption is observed, and at end of every round
        (after a response from A-Bot is observed).
        """
        encState = self.encoder()
        h, c = encState
        return self.featureNet(self.featureNetInputDropout(h[-1]))

    def reinforce(self, reward):
        return self.decoder.reinforce(reward)

