import time
from scipy.special import xlogy
from scipy.sparse import csr_matrix
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join

'''
currently, matrices are X[word][coword] = 1,
so X[word_a][word_b] != X[word_b][word_a], 
since if a word is not included in the window it won't show up;
e.g. if the string is "a b c d e f g" and L = 3,
(d, [a b c d e f g])
but
a won't get a pairing, since it doesn't have a left window; thus
X[d,a] != X[a,d]

this is really the approximate cooccurence, and it's really only 
inaccurate for the first and last 5 words in a document.
'''

cooccurPath = './halfcentury/'

print("Reading dictionaries...")
a = time.time()
with open('./WordToID.p', 'rb') as fp:
    wordToID = pickle.load(fp)
print("Read. Time elapsed: {}".format(time.time() - a))
f = len(wordToID) # length of vocabulary

cooccurFiles = [f for f in listdir(cooccurPath) if isfile(join(cooccurPath, f))]
        
for i in range(len(cooccurFiles)):

    print("Loading a count matrix...")
    a = time.time()
    cFile = cooccurFiles[i]
    with open("./halfcentury/" + cFile, "rb") as fp:
        b = pickle.load(fp)
    print("Loaded. Time elapsed: {}".format(time.time() - a))
    
    print("Tweaking batch...")
    a = time.time()
    X = (b + b.T)/2.0
    print("Tweaked. Time elapsed: {}".format(time.time() - a))

    print("Setting up cooccur...")
    a = time.time()
    S = X.dot(np.ones(f))
    X.setdiag(S)
    trace = np.sum(X.diagonal())
    print("Set up. Time elapsed: {}".format(time.time() - a))

    # now X is the thing we want, discretely. so we have to do the scaling.
    # want log (X[i,j] * trace(X) / X[i]X[j])
    # log(X[i,j]) - log(X[i]) - log(X[j]) + log(trace(X))
    # ETA for xlogy: 9578 seconds = 2h40m

    print("logXij")
    a = time.time()
    logXij = csr_matrix(xlogy(np.sign(X.todense()), X.todense()))
    print(time.time() - a)

    print("logXj")
    a = time.time()
    logXj = csr_matrix(xlogy(np.sign(np.tile(S, [f, 1])), np.tile(S, [f, 1])))
    print(time.time() - a)

    logTraceX = np.log(np.trace(X.todense()))

    print("PMI")
    a = time.time()
    PMI = logXij + logTraceX - logXi - logXi.T
    print(time.time() - a)

    print("PPMI")
    a = time.time()
    PPMI = csr_matrix(np.multiply(PMI.todense(), np.greater(PMI.todense(), np.zeros((f,f)))))
    print(time.time() - a)

    print("Saving to pickle...")
    a = time.time()
    with open('./ppmi/slice{}.p'.format(i), 'wb') as fp:
        pickle.dump(PPMI, fp)
    print("Saved. Time elapsed: {}".format(time.time() - a))