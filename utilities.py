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
import os

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
		pred_file.write(outputLine + "\n")
		gt_file.write(groundLine + "\n")
	pred_file.close()
	gt_file.close()

	BLEUOutput=os.popen("perl multi-bleu.perl -lc " + gt_file_name + " < " + pred_file_name).read()
	return BLEUOutput


################################################################

def sampleFromDistribution(vals):
		p = random.random()
		s=0.0
		for i,v in enumerate(vals):
				s+=v
				if s>=p:
						return i
		return len(vals)-1


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
