import os
from os import listdir
from os.path import isfile, join
import nltk
from collections import Counter
from nltk.corpus import stopwords
import time
import re

textPath = '/Users/bkitano/Desktop/Classes/Spring_2019/thesis/corpus/cleaned_txt/'
onlyfiles = [f for f in listdir(textPath) if isfile(join(textPath, f))]
english_stopwords = stopwords.words('english')

# do it in batches
freqDistsPath = '/Users/bkitano/Desktop/Classes/Spring_2019/thesis/embeddings/freqDists/'

print("starting freq dist batching")

batchSize = 100
batches = [onlyfiles[i:i + batchSize] for i in range(0, len(onlyfiles), batchSize)]
timeElapsed = 0

lastTime = time.time()
for batchIndex in range(1, len(batches)+1):
    
    batch = batches[batchIndex-1]
    print(len(batch))

    # freq = Counter()
    for i in range(len(batch)):
        filename = batch[i]
        try:
            with open( textPath + filename, 'r+') as f:
                rawtext = f.read()
                t = time.time()
                tokens = [t for t in nltk.word_tokenize(rawtext) if t.isalpha() and t.lower() not in english_stopwords]
                print(time.time() - t)
                # subfreq = nltk.FreqDist(tokens)
                # freq += subfreq
                
        except:
            print(filename)
    
    # with open(freqDistsPath + "batch"+ str(batchIndex) + ".csv", "a+") as f:
    #     for k,v in freq.most_common():
    #         f.write( "{}, {}\n".format(k,v) )

    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/batchIndex) * (len(batches) - batchIndex)
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()


# It may be fruitful to develop machine learning approaches to unsupervised learning of spelling variations.
