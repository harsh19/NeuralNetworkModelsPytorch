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
        self.PAD_token= 0
        self.UNK_token = 3
        self.use_cuda = torch.cuda.is_available()
        self.params = params
        self.teacher_forcing_ratio = params.teacher_forcing_ratio

    def performStep(self, input_variable, target_variable, encoder, revcoder, decoder, encoder_optimizer, revcoder_optimizer, decoder_optimizer, criterion, max_target_length, mode="train"):
        
        # mode= {train, inference, decode_with_teacher}
        perform_backprop = (mode=="train")
        calculate_loss = (mode=="train" or mode=="decode_with_teacher")
        use_teacher_forcing = (mode=="decode_with_teacher")
        if mode=="train":
            use_teacher_forcing = True #if random.random() < self.teacher_forcing_ratio else False

        if perform_backprop:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            revcoder_optimizer.zero_grad()

        batch_size = input_variable.size()[0]
        input_length = input_variable.size()[1]  # input -> b,m,1
        if use_teacher_forcing:
            target_length = target_variable.size()[1]
        else:
            target_length = max_target_length

        encoder_hidden = encoder.initHidden(batch_size) # 1,b,h
        revcoder_hidden = revcoder.initHidden(batch_size)
        encoder_outputs = Variable(torch.zeros(input_length, batch_size, encoder.hidden_size)) # m,b,h
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            inp = input_variable[:,ei,:] # b,1
            encoder_output, encoder_hidden = encoder(
                inp, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        for ei in range(input_length):
            inp=input_variable[:,input_length-ei-1,:]
            revcoder_output,revcoder_hidden=revcoder(inp,revcoder_hidden)
            encoder_outputs[input_length-ei-1]=torch.add(encoder_outputs[input_length-ei-1],revcoder_output[0])

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]*batch_size]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        if self.params.cell_type == "lstm":
            encoder_rep = encoder_hidden[0] # hidden is output,state
        else:
            encoder_rep = encoder_hidden

        decoder_hidden =  decoder.initHidden(encoder_rep) #+ revcoder_hidden #encoder_hidden # can also use last encoder output

        if calculate_loss or use_teacher_forcing:
            target_variable = target_variable.squeeze(2) # b,m  . #target_variable: b,m,1

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            total_num_of_tokens = 0
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                #if mode=="train":
                den = sum(target_variable[:,di]!=0)
                den = den.data.cpu().numpy()
                total_num_of_tokens+=den[0]
                cur_batch_loss = criterion(decoder_output, target_variable[:,di]) #/ (1.0 * den[0])
                loss += cur_batch_loss
                decoder_input = target_variable[:,di].unsqueeze(0)  # Teacher forcing b -> 1,b

        else: # Without teacher forcing: use its own predictions as the next input
            
            eos_vals = [0]*batch_size
            total_num_of_tokens = 0
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
                    ret_attention.append(decoder_attention.data.cpu().numpy())
                decoder_input = Variable(torch.LongTensor([ni_vals])).view(1,-1)
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                if calculate_loss:
                    den = sum(target_variable[:,di]!=0)
                    den = den.data.cpu().numpy()
                    total_num_of_tokens+=den[0]
                    cur_batch_loss = criterion(decoder_output, target_variable[:,di])# / (1.0 * den[0])
                    loss += cur_batch_loss
                
                if sum(eos_vals)==batch_size: ## All have reached EOS
                    break

        if calculate_loss:
            loss = loss/total_num_of_tokens
            loss.backward()
            encoder_optimizer.step()
            revcoder_optimizer.step()
            decoder_optimizer.step()

        if mode=="train":
            return loss.data[0]
        elif mode=="decode_with_teacher":
            return ret, ret_attention, loss
        elif mode=="inference":
            return ret, ret_attention, None


    def train(self, training_pairs, criterion, encoder, revcoder, decoder, print_every=10, plot_every=10, learning_rate=0.1, shuffle_batches=True):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        encoder_optimizer=optim.Adam(encoder.parameters())
        decoder_optimizer=optim.Adam(decoder.parameters())
        revcoder_optimizer=optim.Adam(revcoder.parameters())

        evaluate_every = 10

        num_batches = len(training_pairs)
        batch_indices = np.arange(num_batches)
        if shuffle_batches:
            np.random.shuffle( batch_indices )

        totalTokens=0.0
        lossTotal=0.0

        for iter in range(1, num_batches + 1):
            batch_num = batch_indices[iter-1]
            training_pair = training_pairs[batch_num]
            input_variable = training_pair[0]
            target_variable = training_pair[1]
            #print("input_variable = "+str(input_variable.data.size()))

            loss = self.performStep(input_variable, target_variable, encoder, revcoder,decoder, encoder_optimizer, revcoder_optimizer, decoder_optimizer, criterion, self.MAX_LENGTH, mode="train")
            print_loss_total += loss
            plot_loss_total += loss
            lossTotal+=loss
            totalTokens+=(target_variable.size()[0])*(target_variable.size()[1])


            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (utilities.timeSince(start, iter / num_batches),
                                             iter, iter / num_batches * 100, print_loss_avg))
                if print_loss_avg>40:
                    print("Perplexity > exp^40")
                else:
                    print("Perplexity",math.exp(print_loss_avg)) #/totalTokens))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        #utilities.showPlot(plot_losses)


    ######################################################################
    # ==========
   
    # TODO: This does one at a time as of now. Changing to batch should be simple
    # TODO: Uses greedy decoding. Need beam search
    def decode(self, encoder, revcoder, decoder, sentences, max_length, calculate_loss=False, criterion=None):
        decoded_words_all = []
        attentions_numpy_all = []
        use_data_max_out = self.params.use_data_max_out

        for sentence in sentences:
            #print("sentence = "+sentence)
            input_lang, output_lang = self.data_preparer.input_lang, self.data_preparer.output_lang
            inputs = self.data_preparer.variableFromSentence( self.data_preparer.input_lang, sentence, prepend_start_symbol=True )
            if calculate_loss:
                target = self.data_preparer.variableFromSentence( self.data_preparer.output_lang, sentence, prepend_start_symbol=False )
            else:
                target = None
            inputs = inputs.view(1,-1,1)
            batch_size = 1
            input_variable = inputs
            if calculate_loss:
                mode = "decode_with_teacher"
            else:
                mode = "inference"
            outputs, outputs_attention, loss  = self.performStep(input_variable, target, encoder,revcoder,
                             decoder, None, None, None, criterion, max_length, mode=mode)

            # outputs : mout, b
            outputs = np.array(outputs)
            outputs = outputs.transpose(1,0) # b, mout
            decoded_words = []
            for b in range(batch_size): 
                decoded_words.append([])
            for b in range(batch_size):
                cur_outputs = outputs[b]
                if self.params.use_data_max_out:
                    k = len(sentence.split()) #TO DO- Do this using a utility function
                    if len(cur_outputs)>(k+10):
                        cur_outputs=cur_outputs[:k+10]
                        #print("----------using--------")
                for j,output in enumerate(cur_outputs):
                    if output==self.EOS_token:
                        break
                    decoded_words[b].append(output_lang.index2word[output])

            # decoded_words -> b,mout
            # outputs_attention: m_out, b, m_in
            attentions_numpy = np.array(outputs_attention) #.data.numpy() # mout, b, min
            attentions_numpy = np.transpose(attentions_numpy, [1,0,2] ) # mout, b, min -> b,mout,min
            
            # Since batch size is 1
            decoded_words_all.append(decoded_words[0])
            attentions_numpy_all.append(attentions_numpy[0])

        return decoded_words_all, attentions_numpy_all, loss


    def computeAndEvaluateBleu(self, model, data_pairs):
        encoder,revcoder,decoder = model[0], model[1], model[2]
        max_out_length = 20
        sentences = []
        data = data_pairs  #[:10] # change to validation_pairs
        inputs = []
        outputs = []
        for pair in data_pairs:
            inputs.append(pair[0])
            outputs.append(pair[1])
        pred_outputs, attentions, _ = self.decode(encoder,revcoder,decoder,inputs, max_length=self.params.max_out_length, calculate_loss=False)
        pred_outputs = [' '.join(output_words) for output_words in pred_outputs]
        print(len(pred_outputs))
        bleu = utilities.evaluateBleu(pred_outputs, outputs)
        print("BLEU = ", bleu)

    ######################################################################

    def evaluateRandomly(self, encoder,revcoder, decoder, data_pairs, n=10):
        for i in range(n):
            pair = random.choice(data_pairs)
            print('> INPUT (Random from validation): ', pair[0])
            print('= GROUND TRUTH OUTOUT: ', pair[1])
            output_words, attentions, _ = self.decode(encoder,revcoder,decoder, [pair[0]], max_length=self.params.max_out_length, calculate_loss=False)
            output_words, attentions = output_words[0], attentions[0]
            output_sentence = ' '.join(output_words)
            print('< PREDICTED OUTPUT: ', output_sentence)
            print('')


    def evaluateAndShowAttention(self, input_sentence, encoder1,revcoder1,attn_decoder1):
        output_words, attentions, _ = self.decode(
            encoder1,revcoder1,attn_decoder1, [input_sentence], max_length=self.params.max_out_length, calculate_loss=False)
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

            model = models.Model(input_lang.n_words, output_lang.n_words, params)
            encoder1, revcoder1, attn_decoder1 = model.encoder1, model.revcoder1, model.attn_decoder1
            if self.use_cuda:
                encoder1 = encoder1.cuda()
                revcoder1 = revcoder1.cuda()
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
            #Sort training pairs
            self.data_preparer.train_pairs=self.data_preparer.train_pairs[:num_of_data_points]
            self.data_preparer.train_pairs.sort(key = lambda x: len(x[0].split())) #+len(x[1].split()) )
            for i in range(num_batches):
                cur_batch_data = self.data_preparer.train_pairs[i*batch_size:(i+1)*batch_size]
                training_pairs.append( self.data_preparer.variablesFromPairs( cur_batch_data, padding=True ) )
            
            criterion = nn.NLLLoss(ignore_index=0,size_average=False)

            for epoch in range(epochs):
                print("*"*99)
                print("Epoch="+str(epoch))
                self.train(training_pairs, criterion, encoder1, revcoder1, attn_decoder1, print_every=params.print_every_batch)
                print("============= RANDOM VALIDATION")
                self.evaluateRandomly(encoder1, revcoder1, attn_decoder1, n=5, data_pairs=self.data_preparer.valid_pairs)
                print("============= RANDOM TRAINING DECODING")
                self.evaluateRandomly(encoder1, revcoder1, attn_decoder1, n=5, data_pairs=self.data_preparer.train_pairs[:num_of_data_points])

                if params.debug:
                    data_pairs = self.data_preparer.valid_pairs[:20]
                else:
                    data_pairs = self.data_preparer.valid_pairs
                self.computeAndEvaluateBleu( [encoder1, revcoder1, attn_decoder1], data_pairs)

                if epoch%save_every==0:
                    fname = "./tmp/" + self.params.run_name + "_saved_model_" + str(epoch)
                    fname = fname.decode('utf-8')
                    torch.save( {'encoder':encoder1.state_dict(), 'revcoder': revcoder1.state_dict(), 'decoder':attn_decoder1.state_dict()} , fname )

        elif mode=="test":

            data_preparer = pickle.load( open(u'data_preparer.p','rb') )
            self.data_preparer = data_preparer
            input_lang, output_lang, pairs = data_preparer.input_lang, data_preparer.output_lang, \
                data_preparer.test_pairs
            self.MAX_LENGTH = data_preparer.MAX_LENGTH

            hidden_size = params.hidden_size
            embeddings_size = params.embeddings_size # not used as of now
            model = models.Model(input_lang.n_words, output_lang.n_words, hidden_size)
            encoder1, revcoder1, attn_decoder1 = model.encoder1, model.revcoder1, model.attn_decoder1
            if self.use_cuda:
                encoder1 = encoder1.cuda()
                revcoder1 = revcoder1.cuda()
                attn_decoder1 = attn_decoder1.cuda()

            # Load saved model
            fname = "./tmp/saved_model_" + str(2)
            fname = fname.decode('utf-8')
            loaded_model = torch.load( fname )
            encoder1.load_state_dict( loaded_model['encoder'] )
            revcoder1.load_state_dict( loaded_model['revcoder'] )
            attn_decoder1.load_state_dict( loaded_model['decoder'] )

            # bleu on test data  

            # attention on test data         
            output_words, attentions, _ = self.decode(
            encoder1, revcoder1, attn_decoder1, ["je suis trop froid ."], max_length=self.params.max_out_length, calculate_loss=False)
            output_words, attentions = output_words[0], attentions[0]
            plt.matshow(attentions)

            self.evaluateAndShowAttention("elle a cinq ans de moins que moi .", encoder1, revcoder1, attn_decoder1)
            self.evaluateAndShowAttention("elle est trop petit .", encoder1, revcoder1, attn_decoder1)
            self.evaluateAndShowAttention("je ne crains pas de mourir .", encoder1, revcoder1, attn_decoder1)
            self.evaluateAndShowAttention("c est un jeune directeur plein de talent .", encoder1, revcoder1, attn_decoder1)         

        else:

            print("INVALID MODE SELETED")