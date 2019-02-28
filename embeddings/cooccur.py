import re
import os
import time
from nltk import word_tokenize
import pickle
import numpy as np
from os import listdir
from csv import DictReader
from collections import Counter
from os.path import isfile, join
from multiprocessing import Pool
import scipy.sparse as sp

# ---------- CONSTANTS ----------
# read in the freq counter
cooccurPath = '/Users/bkitano/Desktop/Classes/spring_2019/thesis/embeddings/quintile/'

tokensPath = '/Users/bkitano/Desktop/Classes/spring_2019/thesis/corpus/tokens/'

print("------Starting Cooccur------")
print("Loading pickle dicts")
with open('./IDToWord.p', 'rb') as fp:
    IDToWord = pickle.load(fp)
    
with open('./WordToID.p', 'rb') as fp:
    wordToID = pickle.load(fp)

print("Reading in the directory of token files")
rTime = time.time()
tokenFiles = [f for f in listdir(tokensPath) if isfile(join(tokensPath, f))]
print("Read. Elapsed time: {}".format(time.time() - rTime))

# ---------- HELPER FUNCTIONS ------------

# efficient sum of sparse matrices
def sum_sparse(m):
    x = np.zeros(m[0].shape)
    for a in m:
        ri = np.repeat(np.arange(a.shape[0]),np.diff(a.indptr))
        x[ri,a.indices] += a.data
    return x

# ---------- EXECUTOR -----------

# SET PARAMETERS
L = 5 # window size, or L in the paper
P = 7 # number of processors

'''
Logic: each row is a window, each column is a word. So we are vectorizing
each window.
X.T * X = [ window x word ]^T [ window x word ] 
= [ word x window ] [ window x word ]
Say there are V words and W windows
Then X.T * X \in [V x V]
and V[i,j] = \sum_(w in W) (word_i in window w)(word_j in window w)

What are the diagonals? 
https://stackoverflow.com/questions/42814452/co-occurrence-matrix-from-list-of-words-in-python/42814963
'''

def precooccur(tokens):
    a = time.time()
    windowTarget = [tokens[i-L:i + L + 1] for i in range(L, len(tokens) - L)]
    windowIDToWindow = enumerate(windowTarget)
    print(time.time() - a)
    
    windows, words, counts = [], [], []

    b = time.time()
    for windowID, window in windowIDToWindow:
        for word in window:
            if wordToID.get(word) is not None:
                windows.append(windowID)
                words.append(wordToID[word])
                counts.append(1)
    print(time.time() - b)
    
    c = time.time()
    for vword in wordToID.keys():
        windows.append(len(windowTarget)) # a dummy window
        words.append(wordToID[vword])
        counts.append(0)
    print(time.time() - c)
    
    d = time.time()
    X = sp.csr_matrix((counts, (windows, words)))
    print(time.time() - d)
    
    return X
    
def cooccur(tokens):
    # creates word and window pairs to pass to cooccur
    wordTarget = tokens[L:len(tokens) - L]
    windowTarget = [tokens[i-L:i + L + 1] for i in range(L, len(tokens) - L)]
    pairs = list(zip(wordTarget, windowTarget))

    parallelTime = time.time()

    rows, cols, vals = [], [], []
    for vword in wordToID.keys():
        index = wordToID[vword]
        rows.append(index)
        cols.append(index)
        vals.append(0)
        
    for word, window in pairs:
        for coword in window:
            if wordToID.get(word) is not None:
                rows.append(wordToID[word])
                cols.append(wordToID[coword])
                vals.append(1)
                
    X = sp.csr_matrix((vals, (rows, cols)))
    return X

def parseYear(filename):
    year = filename.split('-')[0]
    year = re.sub('\D', '', year)
    return int(year)
# --------- WRITE TOKENS FOR FILES ---------
print("Starting cleaning")

batchCount = 0
running_batch_time = 0
last_time = time.time()
interval = 20
print("creating batches in intervals of {} years".format(interval))

fileYearPairs = [(t, parseYear(t)) for t in tokenFiles if parseYear(t) > 1400]
yearMin = min([fyp[1] for fyp in fileYearPairs])
yearMax = max([fyp[1] for fyp in fileYearPairs])

batches = []
batch = []
batchYear = yearMin
for pair in fileYearPairs:
    filename = pair[0]
    year = pair[1]
    if year - batchYear < interval:
        batch.append(filename)
    else:
        batches.append(batch)
        batchYear = year
        batch = []
        batch.append(filename)
batches.append(batch)
        
print("starting tokens batching")

# need to batch by year
timeElapsed = 0
    
lastTime = time.time()
for batchIndex in range(1, len(batches)+1):
    batch = batches[batchIndex-1]
    
    batchCooccur = sp.csr_matrix((len(wordToID),len(wordToID)))
    for filename in batch:
        with open(tokensPath + filename, 'rb') as fp:
            tokens = pickle.load(fp)
            batchCooccur += cooccur(tokens)
    
#     batchCooccur = np.dot(preCooccur.T, preCooccur)
    
    with open("{}cooccurBatch{}.p".format(cooccurPath, batchIndex), "wb") as fp:
        pickle.dump(batchCooccur, fp)

    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/batchIndex) * (len(batches) - batchIndex)
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()
