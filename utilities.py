import numpy as np
import csv
import configuration as config
import random
import heapq
import re
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
import time

###################################################



def showAttention( input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    #plt.show()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluateBleu(predicted_outputs, ground_truth_outputs):
	pred_file_name = "./tmp/bleu_pred.txt"
	pred_file = open(pred_file_name,"w")
	gt_file_name = "./tmp/bleu_gt.txt"
	gt_file = open(gt_file_name,"w")
	for outputLine,groundLine in zip(predicted_outputs, ground_truth_outputs):
		pred_file.write(outputLine)
		gt_file.write(groundLine)
	pred_file.close()
	gt_file.close()

	BLEUOutput=os.popen("perl multi-bleu.perl -lc " + gt_file_name + " < " + pred_file_name).read()
	return BLEUOutput


################################################################ Beam search data structures
class TopN(object):
	"""Maintains the top n elements of an incrementally provided set."""

	def __init__(self, n):
		self._n = n
		self._data = []

	def size(self):
		assert self._data is not None
		return len(self._data)

	def push(self, x):
		"""Pushes a new element."""
		assert self._data is not None
		if len(self._data) < self._n:
			heapq.heappush(self._data, x)
		else:
			heapq.heappushpop(self._data, x)

	def extract(self, sort=False):
		"""Extracts all elements from the TopN. This is a destructive operation.
		The only method that can be called immediately after extract() is reset().
		Args:
			sort: Whether to return the elements in descending sorted order.
		Returns:
			A list of data; the top n elements provided to the set.
		"""
		assert self._data is not None
		data = self._data
		self._data = None
		if sort:
			data.sort(reverse=True)
		return data

	def reset(self):
		"""Returns the TopN to an empty state."""
		self._data = []

################################################################

class OutputSentence(object):
	"""Represents a complete or partial caption."""

	def __init__(self, sentence, state, logprob, score, metadata=None):
		"""Initializes the Caption.
		Args:
			sentence: List of word ids in the caption.
			state: Model state after generating the previous word.
			logprob: Log-probability of the caption.
			score: Score of the caption.
			metadata: Optional metadata associated with the partial sentence. If not
				None, a list of strings with the same length as 'sentence'.
		"""
		self.sentence = sentence
		self.state = state
		self.logprob = logprob
		self.score = score
		self.metadata = metadata

	def __cmp__(self, other):
		"""Compares Captions by score."""
		assert isinstance(other, OutputSentence)
		if self.score == other.score:
			return 0
		elif self.score < other.score:
			return -1
		else:
			return 1
	
	# For Python 3 compatibility (__cmp__ is deprecated).
	def __lt__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score < other.score
	
	# Also for Python 3 compatibility.
	def __eq__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score == other.score

################################################################

def sampleFromDistribution(vals):
		p = random.random()
		s=0.0
		for i,v in enumerate(vals):
				s+=v
				if s>=p:
						return i
		return len(vals)-1


################################################################
import os
def getBlue(validOutFile_name, original_data_path, BLEUOutputFile_path, decoder_outputs_inference, decoder_ground_truth_outputs, preprocessing_obj, verbose=False):
	validOutFile=open(validOutFile_name,"w")
	for outputLine,groundLine in zip(decoder_outputs_inference, decoder_ground_truth_outputs):
		if verbose:
			print outputLine
		outputLine=preprocessing_obj.fromIdxSeqToVocabSeq(outputLine)
		if "sentend" in outputLine:
			outputLine=outputLine[:outputLine.index("sentend")]
		if verbose:
			print outputLine
			print preprocessing_obj.fromIdxSeqToVocabSeq(groundLine)
		outputLine=" ".join(outputLine)+"\n"
		validOutFile.write(outputLine)
	validOutFile.close()

	BLEUOutput=os.popen("perl multi-bleu.perl -lc " + original_data_path + " < " + validOutFile_name).read()
	BLEUOutputFile=open(BLEUOutputFile_path,"w")
	BLEUOutputFile.write(BLEUOutput)
	BLEUOutputFile.close()



#############################################################




'''
            gt_idx_tmp = self.data_preparer.indexesFromSentence(self.data_preparer.output_lang, pair[1])
            gt_idx_tmp.append(0)
            gt_idx = Variable(torch.LongTensor(gt_idx_tmp).view(1, len(gt_idx_tmp)))
            m = len(gt_idx_tmp)
            oracle = Variable(torch.zeros(m,1,2969))
            for k in range(m):
                oracle[k,0,gt_idx_tmp[k]] = 0.2
                oracle[k,0,1] = 0.8
                print(oracle[k,:,:])
                print(gt_idx[:,k])
                tmp = criterion( oracle[k,:,:] , gt_idx[:,k])
                print("tmp= ",tmp)
                print("------------------------------------------")
            '''
            #pred_idx = self.data_preparer.indexesFromSentence(output_sentence, pair[1])
            