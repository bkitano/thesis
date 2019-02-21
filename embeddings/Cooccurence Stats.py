#!/usr/bin/env python
# coding: utf-8

# We need to make the cooccurrence matrices to then calculate the pointwise mutual information (PMI) values to pass to the Dynamic Word Embedding model. 

from csv import DictReader
from collections import Counter
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
import time

print("Starting...")

# read in the freq counter
totalFreqDistFile = '/Users/bkitano/Desktop/Classes/spring_2019/thesis/embeddings/totalFreqDist.csv'

with open(totalFreqDistFile) as f:
    reader = DictReader(f, fieldnames=['word', 'count'])
    freqDist = Counter({row['word']: int(row['count']) for row in reader})

frequentWords = [w for w in freqDist.keys() if freqDist[w] > 200]

print(len(frequentWords))
print(frequentWords[0:10])

print("Read words...")

# create and save hashmap of word to ids
wordToID = dict(zip(frequentWords, range(len(freqDist))))
IDToWord = dict([(wordToID[word], word) for word in wordToID.keys()])

textPath = '/Users/bkitano/Desktop/Classes/spring_2019/thesis/corpus/cleaned_txt/'

textFiles = [f for f in listdir(textPath) if isfile(join(textPath, f))]

print(len(textFiles))

z = np.zeros((len(frequentWords), len(frequentWords)))
df = pd.DataFrame(z, index = frequentWords, columns = frequentWords)

print(df.shape)

# df['god']
# df['god']['god'] should store the total number of appearances
# df will be a diagonal matrix

cooccurrencePath = '/Users/bkitano/Desktop/Classes/spring_2019/thesis/embeddings/cooccur/'

L = 5 # window size, or L in the paper
# window: x1 x2 ... xL W y1 y2 ... yL

print("Starting co-occurrence...")

lastTime = time.time()
totalTime = 0
batch = 1
batches = 100

with open(textPath + textFiles[0], 'r+') as f:
    text = f.read()
    tokens = [w for w in nltk.word_tokenize(text) if w in frequentWords]
    for i in range(L, len(tokens)-1):
        word = tokens[i]
        window = tokens[i-L : i + L + 1]
        for coword in window:
            try:
                df[word][coword] += 1.0
            except:
                print(word)
                
        if i in range(int(len(tokens)/batches), len(tokens), int(len(tokens)/batches)):
            batchTime = time.time() - lastTime
            totalTime += batchTime
            avgTime = totalTime / batch
            ETA = (batches - batch) * avgTime
            ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

            print( "Batch {} of {} | batch time: {} | ETA : {}".format(batch, batches, batchTime, ETAstring) )
            
            batch += 1
            lastTime = time.time()

