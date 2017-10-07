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
from keras.preprocessing.sequence import pad_sequences

use_cuda = torch.cuda.is_available()

SOS_token = 1
EOS_token = 2
pad_token = 0
unk_token = 3

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2:"EOS", 3:"UNK"}
		self.n_words = 4  # Count SOS and EOS and PAD

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def pruneVocab(self):
		pass #TODO


class Prepro:

	# Turn a Unicode string to plain ASCII, thanks to
	# http://stackoverflow.com/a/518232/2809427

	def unicodeToAscii(self,s):
		return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn'
		)
	# Lowercase, trim, and remove non-letter characters
	def normalizeString(self,s):
		s = self.unicodeToAscii(s.lower().strip())
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
		return s

	def readLangs(self, split='train', reverse=False, do_filtering=True):
		print("Reading lines...")
		src_en = "./data/" + split + ".en-de.low.en" # EN is target
		src_de = "./data/" + split + ".en-de.low.de" # DE is source
		lines_en = open(src_en,"r").readlines()
		lines_de = open(src_de,"r").readlines()
		# Split every line into pairs and normalize
		pairs = [ [self.normalizeString(lines_de[i].strip()), self.normalizeString(lines_en[i].strip())] for i in range(len(lines_en)) ]
		# Reverse pairs, make Lang instances
		if reverse:
			pairs = [list(reversed(p)) for p in pairs]
		pairs = self.filterPairs(pairs)
		print("split = %s. Read %s sentence pairs" % (split, len(pairs)) )
		return pairs


	def filterPair(self, p):
		return len(p[0].split(' ')) < self.MAX_LENGTH and \
			len(p[1].split(' ')) < self.MAX_LENGTH
			# and \p[1].startswith(self.eng_prefixes)

	def filterPairs(self, pairs):
		return [pair for pair in pairs if self.filterPair(pair)]

	def prepareData(self, lang1, lang2, reverse=False):
		if reverse:
			input_lang = Lang(lang2)
			output_lang = Lang(lang1)
		else:
			input_lang = Lang(lang1)
			output_lang = Lang(lang2)
		train_pairs = self.readLangs(reverse=reverse, split="train", do_filtering=True) #do_filtering does length based filtering
		valid_pairs = self.readLangs(reverse=reverse, split="valid", do_filtering=True)
		test_pairs = self.readLangs(reverse=reverse, split="test", do_filtering=True)
		print("Counting words...")
		for pair in train_pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, [train_pairs,valid_pairs,test_pairs]

	def getData(self, max_length):
		self.MAX_LENGTH = max_length
		input_lang, output_lang, pairs = self.prepareData('de', 'eng', reverse=False)
		self.train_pairs, self.valid_pairs, self.test_pairs = pairs
		print(random.choice(self.train_pairs))
		self.input_lang, self.output_lang = input_lang, output_lang
		#return input_lang, output_lang, pairs

	def indexesFromSentence(self, lang, sentence):
		ret = []
		for word in sentence.split(' '):
			#print("word = "+word)
			if word in lang.word2index:
				ret.append(lang.word2index[word] )
			else:
				ret.append(unk_token)
		return ret


	def variableFromSentence(self, lang, sentence):
		indexes = self.indexesFromSentence(lang, sentence)
		indexes.append(EOS_token)
		result = Variable(torch.LongTensor(indexes).view(-1, 1))
		if use_cuda:
			return result.cuda()
		else:
			return result

	def variableFromSentenceBatch(self, lang, sentences, padding):
		all_indexes = []
		batch_size = len(sentences)
		max_len = -1
		for sentence in sentences:
			indexes = self.indexesFromSentence(lang, sentence)
			indexes = indexes + [EOS_token]
			max_len = max(max_len, len(indexes))
		for sentence in sentences:
			indexes = self.indexesFromSentence(lang, sentence)
			indexes = indexes + [EOS_token]
			if len(indexes)<max_len:
				if padding=="post":
					for j in range(max_len-len(indexes)):
						indexes.append(0)
				elif padding=="pre":
					tmp = []
					for j in range(max_len-len(indexes)):
						tmp.append(0)
					for val in indexes: 
						tmp.append(val)
					indexes = tmp
			all_indexes.append(indexes)
		result = Variable(torch.LongTensor(all_indexes).view(batch_size, -1, 1))
		if use_cuda:
			return result.cuda()
		else:
			return result

	def variablesFromPair(self, pair):
		input_lang, output_lang = self.input_lang, self.output_lang
		input_variable = self.variableFromSentence(input_lang, pair[0])
		target_variable = self.variableFromSentence(output_lang, pair[1])
		return (input_variable, target_variable)

	def variablesFromPairs(self, batch_pairs, padding):
		input_lang, output_lang = self.input_lang, self.output_lang
		inp=[]
		out=[]
		max_inp_length = -1
		max_out_length = -1
		for pair in batch_pairs:
			inp.append(pair[0])
			out.append(pair[1])
		
		input_variable = self.variableFromSentenceBatch(input_lang, inp, padding="pre")
		target_variable = self.variableFromSentenceBatch(output_lang, out, padding="post")
		return (input_variable, target_variable)


'''
	def prepareDataEngFra(self, lang1, lang2, reverse=False): #Not used
		input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
		print("Read %s sentence pairs" % len(pairs))
		pairs = self.filterPairs(pairs)
		print("Trimmed to %s sentence pairs" % len(pairs))
		print("Counting words...")
		for pair in pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, pairs

	def readLangs(self,lang1, lang2, reverse=False):
		print("Reading lines...")
		# Read the file and split into lines
		lines = open('data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')
		# Split every line into pairs and normalize
		pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]
		# Reverse pairs, make Lang instances
		if reverse:
			pairs = [list(reversed(p)) for p in pairs]
			input_lang = Lang(lang2)
			output_lang = Lang(lang1)
		else:
			input_lang = Lang(lang1)
			output_lang = Lang(lang2)
		return input_lang, output_lang, pairs
'''