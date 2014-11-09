__author__ = 'Philipp'

import numpy as np
import math as m


def readTermDocument():
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


def readTerms():
    f = open("data/bbcnews.terms")
    ret = []
    for line in f.readlines():
        ret.append(line.replace("\n", ""))
    f.close()
    return ret


def tfIdf(A):
    sumTerms = [sum(colum) for colum in A.T]  # sum of all terms in a document
    sumWord = [sum(x >= 1 for x in row) for row in A]
    numDoc = len(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i, j] = (A[i, j] / sumTerms[j]) * m.log(numDoc / sumWord[i])
    return A


A = readTermDocument()
print(A.shape)
print(len(readTerms()))
print(A.max())
A = tfIdf(A)
print(A.max())
