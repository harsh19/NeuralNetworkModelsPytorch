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

SOS_token = 0
EOS_token = 1

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

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


class Prepro:

	# Turn a Unicode string to plain ASCII, thanks to
	# http://stackoverflow.com/a/518232/2809427

	def __init__(self):
		self.MAX_LENGTH = 10
		self.eng_prefixes = (
	"i am ", "i m ",
	"he is", "he s ",
	"she is", "she s",
	"you are", "you re ",
	"we are", "we re ",
	"they are", "they re ")


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

	def readLangs(self,lang1, lang2, reverse=False):
		print("Reading lines...")
		# Read the file and split into lines
		#lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
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

	def filterPair(self, p):
		return len(p[0].split(' ')) < self.MAX_LENGTH and \
			len(p[1].split(' ')) < self.MAX_LENGTH and \
			p[1].startswith(self.eng_prefixes)

	def filterPairs(self, pairs):
		return [pair for pair in pairs if self.filterPair(pair)]

	def prepareData(self, lang1, lang2, reverse=False):
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

	def getData(self):
		input_lang, output_lang, pairs = self.prepareData('eng', 'fra', True)
		print(random.choice(pairs))
		self.input_lang, self.output_lang, self.pairs = input_lang, output_lang, pairs
		return input_lang, output_lang, pairs


	def indexesFromSentence(self, lang, sentence):
		return [lang.word2index[word] for word in sentence.split(' ')]


	def variableFromSentence(self, lang, sentence):
		indexes = self.indexesFromSentence(lang, sentence)
		indexes.append(EOS_token)
		result = Variable(torch.LongTensor(indexes).view(-1, 1))
		if use_cuda:
			return result.cuda()
		else:
			return result


	def variablesFromPair(self, pair):
		input_lang, output_lang, pairs = self.input_lang, self.output_lang, self.pairs
		input_variable = self.variableFromSentence(input_lang, pair[0])
		target_variable = self.variableFromSentence(output_lang, pair[1])
		return (input_variable, target_variable)

