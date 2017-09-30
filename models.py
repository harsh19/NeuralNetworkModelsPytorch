from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np
import pickle
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #print("forward: " + str(input.data.shape))
        embedded = self.embedding(input).view(1, 5, -1)
        output = embedded
        print("forward: --hidden: " + str(hidden.data.size()))
        print("forward: embedded: " + str(embedded.data.size()))
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        print("forward: output: " + str(output.data.size()))
        print("forward: hidden: " + str(hidden.data.size()))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size)) # seq length, batchsize, hidden size
        if use_cuda:
            return result.cuda()
        else:
            return result



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        print("forward: " + str(output.data.shape))
        print("forward: " + str(self.embedding(input).data.shape))
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) 
        #nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def computeAttention(self, previous_hidden, encoder_outputs):
        # encoder_outputs: m, b, h
        query = self.attn(previous_hidden) # 1,b,h
        print("query: " + str(query.data.size()))  
        #print("torch.transpose(query, 1, 2) : " + str(torch.transpose(query, 1, 2).data.size()))  
        attn_weights = torch.bmm( torch.transpose(encoder_outputs,0,1), torch.transpose(torch.transpose(query, 0, 1), 1,2) ) # b,m,h x b,h,1 => b,m,1
        print("attn_weights: " + str(attn_weights.data.size()))  
        attn_weights = attn_weights.squeeze(2) # b,m,1 => b,m
        attn_weights = F.softmax(attn_weights) # softmax
        print("attn_weights after softmax: " + str(attn_weights.data.size()))  
        context = torch.bmm(attn_weights.unsqueeze(1),
                                 torch.transpose(encoder_outputs,0,1) ) # b,m->b,1,m  m,b,h -> b,m,h  : b,1,m x b,m,h => b,1,h
        context = context.squeeze() # b,h
        print("context: " + str(context.data.size()))     
        return context, attn_weights


    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input) #.view(1, 5, -1)
        embedded = self.dropout(embedded)
        #print("========================")

        print("input: " + str(input.data.shape)) #1,b
        print("forward emb: " + str(embedded.data.shape)) # 1,b,es
        #print("forward: " + str(self.embedding(input).data.size()))
        print("encoder_outputs: " + str(encoder_outputs.data.size())) # 10,b,h_enc
        print("hidden: " + str(hidden.data.size())) # 1,b,h_dec
        

        '''attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        print("attn_weights: " + str(attn_weights.data.size()))        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        print("attn_applied: " + str(attn_applied.data.size())) 
        '''       
        context, attn_weights = self.computeAttention(hidden, encoder_outputs)

        output = torch.cat((embedded[0], context), 1) # b,2*h
        output = self.attn_combine(output) # b,2h x 2h,h => b,h
        output = output.unsqueeze(0) # 1,b,h
        print("output after context embedding mix: " + str(output.data.size())) # 1,b,h_dec


        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        print("output after softmax: " + str(output.data.size())) # 1,b,h_dec

        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



