# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np
import pickle
import json
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker

import prepro
from models import EncoderRNN
from models import AttnDecoderRNN
import utilities
import models

print("cuda available: " + str(torch.cuda.is_available()) )

class Solver:

    def __init__(self, params):
        self.SOS_token = 1
        self.EOS_token = 2
        self.use_cuda = torch.cuda.is_available()
        self.params = params
        self.teacher_forcing_ratio = params.teacher_forcing_ratio

    def performStep(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_target_length, mode="train"):

        if mode=="train":
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

        batch_size = input_variable.size()[0]
        input_length = input_variable.size()[1]  # input -> b,m,1
        if mode=="train":
            target_length = target_variable.size()[1]
        else:
            target_length = max_target_length

        encoder_hidden = encoder.initHidden(batch_size) # 1,b,h
        encoder_outputs = Variable(torch.zeros(input_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        #print("::: ",input_variable.size())
        for ei in range(input_length):
            inp = input_variable[:,ei,:] # b,1
            encoder_output, encoder_hidden = encoder(
                inp, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]
        decoder_input = Variable(torch.LongTensor([[self.SOS_token]*batch_size]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        
        decoder_hidden = encoder_hidden # can also use last encoder output

        if mode=="train":
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False
        if mode=="train":
            target_variable = target_variable.squeeze(2) # b,m  . #target_variable: b,m,1

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                if mode=="train":
                    cur_batch_loss = (criterion(decoder_output, target_variable[:,di]))/(1.0*batch_size) 
                    loss += cur_batch_loss
                    #print(cur_batch_loss)
                decoder_input = target_variable[:,di].unsqueeze(0)  # Teacher forcing b -> 1,b

        else: # Without teacher forcing: use its own predictions as the next input
            
            eos_vals = [0]*batch_size
            if mode=="inference":
                ret = []
                ret_attention = []
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                ni_vals = []
                for i,output in enumerate(decoder_output):
                    topv, topi = output.data.topk(1)
                    ni = topi[0] # greedy decoding
                    ni_vals.append(ni)
                    if ni==self.EOS_token:
                        eos_vals[i] = 1
                if mode=="inference":
                    ret.append(ni_vals)
                    ret_attention.append(decoder_attention.data.numpy())
                decoder_input = Variable(torch.LongTensor([ni_vals]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                if mode=="train":
                    den = sum( target_variable[:,di] == 0 )
                    cur_batch_loss = (criterion(decoder_output, target_variable[:,di]))/( 1.0*(batch_size) ) 
                    loss += cur_batch_loss
                if sum(ni_vals)==batch_size: ## All have reached EOS
                    break

        if mode=="train":
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            return loss.data[0]
        else:
            return ret, ret_attention


    def train(self, training_pairs, criterion, encoder, decoder, print_every=1000, plot_every=100, learning_rate=0.1, shuffle_batches=True):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        evaluate_every = 10

        num_batches = len(training_pairs)
        batch_indices = np.arange(num_batches)
        if shuffle_batches:
            np.random.shuffle( batch_indices )
        for iter in range(1, num_batches + 1):
            batch_num = batch_indices[iter-1]
            training_pair = training_pairs[batch_num]
            input_variable = training_pair[0]
            target_variable = training_pair[1]
            #print("input_variable = "+str(input_variable.data.size()))

            loss = self.performStep(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, self.MAX_LENGTH, mode="train")
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (utilities.timeSince(start, iter / num_batches),
                                             iter, iter / num_batches * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        #utilities.showPlot(plot_losses)


    ######################################################################
    # ==========
   
    # TODO: This does one at a time as of now. Changing to batch should be simple
    # TODO: Uses greedy decoding. Need beam search
    def decode(self, encoder, decoder, sentences, max_length):
        decoded_words_all = []
        attentions_numpy_all = []

        for sentence in sentences:
            #print("sentence = "+sentence)
            input_lang, output_lang = self.data_preparer.input_lang, self.data_preparer.output_lang
            inputs = self.data_preparer.variableFromSentence( self.data_preparer.input_lang, sentence )
            inputs = inputs.view(1,-1,1)
            batch_size = 1
            input_variable = inputs
            outputs, outputs_attention = self.performStep(input_variable, None, encoder,
                             decoder, None, None, None, self.MAX_LENGTH, mode="inference")
            # outputs_attention: m_out, b, m_in
            decoded_words = []
            for b in range(batch_size): decoded_words.append([])
            j=0
            for outputs_at_idx in outputs:
                for idx,output in enumerate(outputs_at_idx):
                    if output==2: #TODO: use language index to find out index of EOS token
                        break
                    decoded_words[idx].append( output_lang.index2word[output] )
                j+=1
            # decoded_words -> b,mout
            #print("decoded_words= "+str(decoded_words))
            attentions_numpy = np.array(outputs_attention) #.data.numpy() # mout, b, min
            attentions_numpy = np.transpose(attentions_numpy, [1,0,2] ) # mout, b, min -> b,mout,min
            
            # Since batch size is 1
            decoded_words_all.append(decoded_words[0])
            attentions_numpy_all.append(attentions_numpy[0])

        return decoded_words_all, attentions_numpy_all


    def computeAndEvaluateBleu(self, model, data_pairs):
        encoder, decoder = model[0], model[1]
        max_out_length = 20
        sentences = []
        data = data_pairs  #[:10] # change to validation_pairs
        inputs = []
        outputs = []
        for pair in data_pairs:
            inputs.append(pair[0])
            outputs.append(pair[1])
        pred_outputs, attentions = self.decode(encoder, decoder, inputs[:10], max_length=10)
        pred_outputs = [' '.join(output_words) for output_words in pred_outputs]
        print(len(pred_outputs))
        bleu = utilities.evaluateBleu(pred_outputs, outputs)
        print("BLEU = ", bleu)

    ######################################################################

    def evaluateRandomly(self, encoder, decoder, n=10, lim=-1, criterion=None):
        for i in range(n):
            pair = random.choice(self.data_preparer.valid_pairs[:lim])
            print('> INPUT (Random from validation): ', pair[0])
            print('= GROUND TRUTH OUTOUT: ', pair[1])
            output_words, attentions = self.decode(encoder, decoder, [pair[0]], max_length=10)
            output_words, attentions = output_words[0], attentions[0]
            output_sentence = ' '.join(output_words)
            print('< PREDICTED OUTPUT: ', output_sentence)
            print('')


    def evaluateAndShowAttention(self, input_sentence, encoder1, attn_decoder1):
        output_words, attentions = self.decode(
            encoder1, attn_decoder1, [input_sentence], max_length=10)
        output_words, attentions = output_words[0], attentions[0]
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        utilities.showAttention(input_sentence, output_words, attentions)

    ######################################################################


    def main(self):

        params = self.params
        mode = params.mode

        ## prepro
        if mode=="prepro":
            self.data_preparer = data_preparer = prepro.Prepro()
            data_preparer.getData(params.max_length)
            pickle.dump(data_preparer, open(u'data_preparer.p','wb'))
        
        elif mode=="train":
            
            data_preparer = pickle.load( open(u'data_preparer.p','rb') )
            self.data_preparer = data_preparer
            input_lang, output_lang, pairs = data_preparer.input_lang, data_preparer.output_lang, \
                data_preparer.train_pairs
            self.MAX_LENGTH = data_preparer.MAX_LENGTH

            hidden_size = params.hidden_size
            embeddings_size = params.embeddings_size # not used as of now
            model = models.Model(input_lang.n_words, output_lang.n_words, hidden_size)
            encoder1, attn_decoder1 = model.encoder1, model.attn_decoder1
            if self.use_cuda:
                encoder1 = encoder1.cuda()
                attn_decoder1 = attn_decoder1.cuda()

            epochs = params.epochs
            num_of_data_points = params.num_of_points #9600
            if num_of_data_points==-1:
                num_of_data_points = len(self.data_preparer.train_pairs)
            print("POINTS BEING USED FOR TRAINING / TOTAL DATA POINTS",num_of_data_points, len(self.data_preparer.train_pairs))
            batch_size = params.batch_size
            save_every = params.save_every_epoch

            training_pairs = []
            num_batches = int(num_of_data_points / batch_size)
            for i in range(num_batches):
                cur_batch_data = self.data_preparer.train_pairs[i*batch_size:(i+1)*batch_size]
                training_pairs.append( self.data_preparer.variablesFromPairs( cur_batch_data, padding=True ) )
            
            criterion = nn.NLLLoss(ignore_index=0)

            for epoch in range(epochs):
                print("*"*99)
                print("Epoch="+str(epoch))
                self.train(training_pairs, criterion, encoder1, attn_decoder1, print_every=10)
                self.evaluateRandomly(encoder1, attn_decoder1, n=5, criterion=criterion)
                self.computeAndEvaluateBleu( [encoder1, attn_decoder1], self.data_preparer.valid_pairs)
                if epoch%save_every==0:
                    fname = "./tmp/saved_model_" + str(epoch)
                    fname = fname.decode('utf-8')
                    torch.save( {'encoder':encoder1.state_dict(), 'decoder':attn_decoder1.state_dict()} , fname )

        elif mode=="test":

            data_preparer = pickle.load( open(u'data_preparer.p','rb') )
            self.data_preparer = data_preparer
            input_lang, output_lang, pairs = data_preparer.input_lang, data_preparer.output_lang, \
                data_preparer.test_pairs
            self.MAX_LENGTH = data_preparer.MAX_LENGTH

            hidden_size = params.hidden_size
            embeddings_size = params.embeddings_size # not used as of now
            model = models.Model(input_lang.n_words, output_lang.n_words, hidden_size)
            encoder1, attn_decoder1 = model.encoder1, model.attn_decoder1
            if self.use_cuda:
                encoder1 = encoder1.cuda()
                attn_decoder1 = attn_decoder1.cuda()

            # Load saved model
            fname = "./tmp/saved_model_" + str(2)
            fname = fname.decode('utf-8')
            loaded_model = torch.load( fname )
            encoder1.load_state_dict( loaded_model['encoder'] )
            attn_decoder1.load_state_dict( loaded_model['decoder'] )

            # bleu on test data  

            # attention on test data         
            output_words, attentions = self.decode(
            encoder1, attn_decoder1, ["je suis trop froid ."], max_length=10)
            output_words, attentions = output_words[0], attentions[0]
            plt.matshow(attentions)

            self.evaluateAndShowAttention("elle a cinq ans de moins que moi .", encoder1, attn_decoder1)
            self.evaluateAndShowAttention("elle est trop petit .", encoder1, attn_decoder1)
            self.evaluateAndShowAttention("je ne crains pas de mourir .", encoder1, attn_decoder1)
            self.evaluateAndShowAttention("c est un jeune directeur plein de talent .", encoder1, attn_decoder1)         

        else:

            print("INVALID MODE SELETED")
