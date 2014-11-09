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


def gettesta():
    return np.array(((3.0, 4.0, 0), (17.0, 0.0, 1.0), (0.0, 2.0, 3.0)))


def readterms():
    f = open("data/bbcnews.terms")
    ret = []
    for line in f.readlines():
        ret.append(line.replace("\n", ""))
    f.close()
    return ret


def tfidf(a):
    sumTerms = [sum(colum) for colum in a.T]  # sum of all terms in a document
    sumWord = [sum(x >= 1 for x in row) for row in a]
    numDoc = len(a)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i, j] = (a[i, j] / sumTerms[j]) * m.log(numDoc / sumWord[i])
    return a


def test_tfidf():
    testmatrix = gettesta()
    print(testmatrix)
    print(tfidf(testmatrix))


def initWH(a, k):
    return np.random.random_sample((a.shape[0], k)), np.random.random_sample((k, a.shape[1]))


def iterstep(a, w, h):
    wh = np.dot(w, h)
    wwh = np.array(np.mat(w.T) * wh)
    wa = np.array(np.mat(w.T) * a)
    h = h * (wa / wwh)
    return h


print(iterstep(gettesta(), np.array(((1, 1), (1, 1), (1, 1))), np.array(((2, 2, 2), (2, 2, 2)))))
A = gettesta()
'''print(A.shape)
print(len(readterms()))
print(A.max())
# A = tfidf(A)
# print(A.max())
wh = initWH(A, 2)
print(wh[0].shape)
print(wh[1].shape)
'''
