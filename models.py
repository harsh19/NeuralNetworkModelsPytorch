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

class Model:
    def __init__(self, inp_words, output_words, hidden_size, n_layers=1):
        self.encoder1 = encoder1 = EncoderRNN(inp_words, hidden_size)
        self.attn_decoder1 = attn_decoder1 = AttnDecoderRNN(hidden_size, output_words,1, dropout_p=0.1)
        #return encoder1, attn_decoder1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        batch_size = input.size()[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) 
        #nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def computeAttention(self, previous_hidden, encoder_outputs):
        # encoder_outputs: m, b, h
        query = previous_hidden # self.attn(previous_hidden) # 1,b,h
        attn_weights = torch.bmm( torch.transpose(encoder_outputs,0,1), torch.transpose(torch.transpose(query, 0, 1), 1,2) ) # b,m,h x b,h,1 => b,m,1
        attn_weights = attn_weights.squeeze(2) # b,m,1 => b,m
        attn_weights = F.softmax(attn_weights) # softmax
        #print("attn_weights after softmax: " + str(attn_weights.data.size()))  
        context = torch.bmm(attn_weights.unsqueeze(1),
                                 torch.transpose(encoder_outputs,0,1) ) # b,m->b,1,m  m,b,h -> b,m,h  : b,1,m x b,m,h => b,1,h
        context = context.squeeze(1) # b,h
        return context, attn_weights


    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input) #.view(1, 5, -1)
        embedded = self.dropout(embedded)
        context, attn_weights = self.computeAttention(hidden, encoder_outputs)
        output = torch.cat((embedded[0], context), 1) # b,2*h
        output = self.attn_combine(output) # b,2h x 2h,h => b,h
        output = output.unsqueeze(0) # 1,b,h
        #print("output after context embedding mix: " + str(output.data.size())) # 1,b,h_dec

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        #print("output after softmax: " + str(output.data.size())) # b,vocab_size
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



