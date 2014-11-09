__author__ = 'Philipp'

import math as m

import numpy as np


def readtermdocument():
    f = open("data/bbcnews.mtx")
    counter = 0
    for line in f.readlines():
        if counter == 1:
            debug = tuple([int(elem) for elem in line.split(" ", 2)[:2]])
            ret = np.zeros(debug, dtype=float)
        else:
            if counter > 1:
                vals = line.replace("\n", "").split(' ', 2)
                x, y = map(int, vals[:2])
                z = float(vals[2])
                ret[x - 1, y - 1] = z
        counter += 1
    f.close()
    return ret


def readterms():
    f = open("data/bbcnews.terms")
    ret = []
    for line in f.readlines():
        ret.append(line.replace("\n", ""))
    f.close()
    return ret


def tfidf(A):
    sumTerms = [sum(colum) for colum in A.T]  # sum of all terms in a document
    sumWord = [sum(x >= 1 for x in row) for row in A]
    numDoc = len(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i, j] = (A[i, j] / sumTerms[j]) * m.log(numDoc / sumWord[i])
    return A


def test_tfidf():
    testMatrix = np.array(((3.0, 4.0, 0), (17.0, 0.0, 1.0), (0.0, 2.0, 3.0)))
    print(testMatrix)
    print(tfidf(testMatrix))


def initWH(A, k):
    return np.random.random_sample((A.shape[0], k)), np.random.random_sample((k, A.shape[1]))


A = readtermdocument()
print(A.shape)
print(len(readterms()))
print(A.max())
# A = tfidf(A)
#print(A.max())
wh = initWH(A, 2)
print(wh[0].shape)
print(wh[1].shape)
