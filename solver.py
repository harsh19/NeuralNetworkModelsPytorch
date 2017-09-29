# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np
import configuration as config
import pickle
import json
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


import prepro
from models import EncoderRNN
from models import AttnDecoderRNN

class Solver:

    def __init__(self):
        self.SOS_token = 0
        self.EOS_token = 1
        self.use_cuda = torch.cuda.is_available()
        self.teacher_forcing_ratio = 0.5


    def train(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))


    def trainIters(self, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [ self.data_preparer.variablesFromPair(random.choice(self.data_preparer.pairs))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = self.train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, self.MAX_LENGTH)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        self.showPlot(plot_losses)


    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)


    ######################################################################
    # Evaluation
    # ==========
    #
    # Evaluation is mostly the same as training, but there are no targets so
    # we simply feed the decoder's predictions back to itself for each step.
    # Every time it predicts a word we add it to the output string, and if it
    # predicts the EOS token we stop there. We also store the decoder's
    # attention outputs for display later.
    #

    def evaluate(self, encoder, decoder, sentence, max_length):
        input_variable = variableFromSentence(input_lang, sentence)
        input_length = input_variable.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni])
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        return decoded_words, decoder_attentions[:di + 1]


    ######################################################################
    # We can evaluate random sentences from the training set and print out the
    # input, target, and output to make some subjective quality judgements:
    #

    def evaluateRandomly(self, encoder, decoder, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')




    ######################################################################
    # For a better viewing experience we will do the extra work of adding axes
    # and labels:
    #

    def showAttention(input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()


    def evaluateAndShowAttention(input_sentence):
        output_words, attentions = evaluate(
            encoder1, attn_decoder1, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)

    def main(self):

        ## prepro
        self.data_preparer = data_preparer = prepro.Prepro()
        input_lang, output_lang, pairs = data_preparer.getData()
        print("="*50)
        self.MAX_LENGTH = data_preparer.MAX_LENGTH
        print("="*50)

        ######################################################################
        # Training and Evaluating
        # =======================
        #
        # With all these helper functions in place (it looks like extra work, but
        # it's easier to run multiple experiments easier) we can actually
        # initialize a network and start training.
        #
        # Remember that the input sentences were heavily filtered. For this small
        # dataset we can use relatively small networks of 256 hidden nodes and a
        # single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
        # reasonable results.
        #
        # .. Note:: 
        #    If you run this notebook you can train, interrupt the kernel,
        #    evaluate, and continue training later. Comment out the lines where the
        #    encoder and decoder are initialized and run ``trainIters`` again.
        #

        hidden_size = 256
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                       1, dropout_p=0.1, max_length=self.MAX_LENGTH)

        if self.use_cuda:
            encoder1 = encoder1.cuda()
            attn_decoder1 = attn_decoder1.cuda()

        self.trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

        ######################################################################
        #

        self.evaluateRandomly(encoder1, attn_decoder1)


        ######################################################################
        # Visualizing Attention
        # ---------------------
        #
        # A useful property of the attention mechanism is its highly interpretable
        # outputs. Because it is used to weight specific encoder outputs of the
        # input sequence, we can imagine looking where the network is focused most
        # at each time step.
        #
        # You could simply run ``plt.matshow(attentions)`` to see attention output
        # displayed as a matrix, with the columns being input steps and rows being
        # output steps:
        #

        output_words, attentions = self.evaluate(
            encoder1, attn_decoder1, "je suis trop froid .")
        plt.matshow(attentions.numpy())

        self.evaluateAndShowAttention("elle a cinq ans de moins que moi .")

        self.evaluateAndShowAttention("elle est trop petit .")

        self.evaluateAndShowAttention("je ne crains pas de mourir .")

        self.evaluateAndShowAttention("c est un jeune directeur plein de talent .")


        ######################################################################
        # Exercises
        # =========
        #
        # -  Try with a different dataset
        #
        #    -  Another language pair
        #    -  Human → Machine (e.g. IOT commands)
        #    -  Chat → Response
        #    -  Question → Answer
        #
        # -  Replace the embeddings with pre-trained word embeddings such as word2vec or
        #    GloVe
        # -  Try with more layers, more hidden units, and more sentences. Compare
        #    the training time and results.
        # -  If you use a translation file where pairs have two of the same phrase
        #    (``I am test \t I am test``), you can use this as an autoencoder. Try
        #    this:
        #
        #    -  Train as an autoencoder
        #    -  Save only the Encoder network
        #    -  Train a new Decoder for translation from there
        #
