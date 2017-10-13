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
    def __init__(self, inp_words, output_words, params, n_layers=1):
        self.encoder1 = encoder1 = EncoderRNN(inp_words, params.enc_hidden_size, embedding_size=params.enc_embedding_size, cell_type=params.cell_type)
        self.revcoder1 = revcoder1 = EncoderRNN(inp_words, params.enc_hidden_size, embedding_size=params.enc_embedding_size, share_embeddings=True, reference_embeddings=self.encoder1.embedding, cell_type=params.cell_type)
        self.attn_decoder1 = attn_decoder1 = AttnDecoderRNN(params.dec_hidden_size, output_words, params.dec_embedding_size, context_size=params.enc_hidden_size ,dropout_p=params.dec_dropout, cell_type=params.cell_type)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1, share_embeddings=False, reference_embeddings=None, cell_type="gru"):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if share_embeddings:
            self.embedding = reference_embeddings
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)
        if cell_type=="gru":
            self.rnn = nn.GRU(embedding_size, hidden_size) ## TODO: embedding size
        elif cell_type=="lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size)
        self.cell_type=cell_type

    def forward(self, input, hidden):
        batch_size = input.size()[0]
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        #print("output=",output.data.size())
        #print("hidden=",hidden.data.size())
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        #result = Variable(torch.zeros(1, batch_size, self.hidden_size)) # seq length, batchsize, hidden size
        if self.cell_type=="gru":
            result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        elif self.cell_type=="lstm":
            result = ( Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size)))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size, context_size ,n_layers=1,dropout_p=0.1, cell_type="gru"):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size) 
        #nn.Linear(self.hidden_size * 2, self.max_length)
        #self.attn_combine = nn.Linear(context_size + embedding_size, embedding_size)
        if dropout_p>=0:
            self.dropout = nn.Dropout(self.dropout_p)
        if cell_type=="gru":
            self.rnn = nn.GRU(embedding_size + context_size, hidden_size) #TODO: Use embedding size
        elif cell_type=="lstm":
            self.rnn = nn.LSTM(embedding_size + context_size, hidden_size)
        self.cell_type = cell_type
        self.out = nn.Linear(hidden_size + context_size, output_size)

    def computeAttention(self, previous_hidden, encoder_outputs): # encoder_outputs: m, b, h
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
        if self.dropout_p>=0:
            embedded = self.dropout(embedded)
            print("======= using dropout========")
        if self.cell_type=="lstm":
            query = hidden[0] # hidden = output, cell.
        else:
            query = hidden
        context, attn_weights = self.computeAttention(query, encoder_outputs)
        output = torch.cat((embedded[0], context), 1) # b,emb_dec+h_enc
        #output = self.attn_combine(output) # b,(emb_dec+h_enc) x (h_dec+h_enc),emb_dec => b,emb_dec
        output = output.unsqueeze(0) # 1,b,emb_dec+h_enc

        for i in range(self.n_layers): # assuming only 1 layer
            #output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        input_to_classifier = torch.cat((output[0], context), 1) # b,2*h
        output = F.log_softmax(self.out(input_to_classifier))
        return output, hidden, attn_weights

    def initHidden(self, encoder_hidden):
        if self.cell_type=="gru":
            result = encoder_hidden
        else:
            result = (encoder_hidden,encoder_hidden)
        if use_cuda:
            return result.cuda()
        else:
            return result
        


