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


import numpy as np


import torch


from torch.autograd import Variable


import torch.nn.functional as F


from torch.utils.data import DataLoader


import random


import torch.nn as nn


import torch.optim as optim


import math


from torch.nn.utils.rnn import pack_padded_sequence


from torch.distributions import Categorical


class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()


class Decoder(nn.Module):

    def __init__(self, vocabSize, embedSize, rnnHiddenSize, numLayers,
        startToken, endToken, dropout=0, **kwargs):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.startToken = startToken
        self.endToken = endToken
        self.dropout = dropout
        self.rnn = nn.LSTM(self.embedSize, self.rnnHiddenSize, self.
            numLayers, batch_first=True, dropout=self.dropout)
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

    def forwardDecode(self, encStates, maxSeqLen=20, inference='sample',
        beamSize=1):
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
                raise ValueError("Invalid inference type: '{}'".format(
                    inference))
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
        encStates = [x.unsqueeze(2).repeat(1, 1, numOptions, 1).view(self.
            numLayers, -1, self.rnnHiddenSize) for x in encStates]
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
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(self
            .endToken)
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
                hiddenStates = [x.unsqueeze(2).repeat(1, 1, beamSize, 1) for
                    x in hiddenStates]
                hiddenStates = [x.view(self.numLayers, -1, self.
                    rnnHiddenSize) for x in hiddenStates]
            else:
                emb = self.wordEmbed(beamTokensTable[:, :, (t - 1)])
                output, hiddenStates = self.rnn(emb.view(-1, 1, self.
                    embedSize), hiddenStates)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                    self.vocabSize)
                if LENGTH_NORM:
                    logProbs = logProbsCurrent * (aliveVector.float() / (t + 1)
                        )
                    coeff_ = aliveVector.eq(0).float() + aliveVector.float(
                        ) * t / (t + 1)
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    logProbs = logProbsCurrent * aliveVector.float()
                    logProbs += logProbSums.unsqueeze(2)
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :, (0)] = 0
                minus_infinity_ = torch.min(logProbs).data[0]
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)
                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(
                    batchSize, beamSize, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.
                    vocabSize).view(batchSize, -1)
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                logProbSums = topLogProbs
                beamTokensTable[:, :, (t)] = tokensArray.gather(1, topIdx)
                backIndices[:, :, (t)] = backIndexArray.gather(1, topIdx)
                hiddenCurrent, cellCurrent = hiddenStates
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(num_layers, batchSize,
                    beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(num_layers, batchSize,
                    beamSize, rnnHiddenSize)
                backIndexVector = backIndices[:, :, (t)].unsqueeze(0
                    ).unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
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
            tokens.append(beamTokensTable[:, :, (tokenIdx)].gather(1,
                backID).unsqueeze(2))
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

    def __init__(self, vocabSize, embedSize, rnnHiddenSize, numLayers,
        useIm, imgEmbedSize, imgFeatureSize, numRounds, isAnswerer, dropout
        =0, startToken=None, endToken=None, **kwargs):
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
        self.wordEmbed = nn.Embedding(self.vocabSize, self.embedSize,
            padding_idx=0)
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
            self.quesRNN = nn.LSTM(quesInputSize, self.rnnHiddenSize, self.
                numLayers, batch_first=True, dropout=0)
        self.factRNN = nn.LSTM(self.embedSize, self.rnnHiddenSize, self.
            numLayers, batch_first=True, dropout=0)
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

    def observe(self, round, image=None, caption=None, ques=None, ans=None,
        captionLens=None, quesLens=None, ansLens=None):
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
            self.questionEmbeds.append(self.wordEmbed(self.questionTokens[idx])
                )
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        """Embed facts i.e. caption and round 0 or question-answer pair otherwise"""
        if factIdx == 0:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(self.factRNN, seq, seqLens,
                returnStates=True)
        elif factIdx > 0:
            quesTokens, quesLens = self.questionTokens[factIdx - 1
                ], self.questionLens[factIdx - 1]
            ansTokens, ansLens = self.answerTokens[factIdx - 1
                ], self.answerLengths[factIdx - 1]
            qaTokens = utils.concatPaddedSequences(quesTokens, quesLens,
                ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(self.factRNN, qa, qaLens,
                returnStates=True)
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
        qEmbed, states = utils.dynamicRNN(self.quesRNN, quesIn, quesLens,
            returnStates=True)
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_batra_mlp_lab_visdial_rl(_paritybench_base):
    pass
