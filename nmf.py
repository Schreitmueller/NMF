__author__ = 'Philipp Schreitmueller'

import math as m

import numpy as np


def read_term_document():
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


def read_terms():
    f = open("data/bbcnews.terms")
    ret = []
    for line in f.readlines():
        ret.append(line.replace("\n", ""))
    f.close()
    return ret


def tf_idf(a):
    sumTerms = [sum(colum) for colum in a.T]  # sum of all terms in a document
    sumWord = [sum(x >= 1 for x in row) for row in a]
    numDoc = len(a)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i, j] = (a[i, j] / sumTerms[j]) * m.log(numDoc / sumWord[i])
    return a


def init_wh(a, k):
    return np.ones((a.shape[0], k)), np.ones((k, a.shape[1]))


def iter_step(a, w, h):
    wh = np.dot(w, h)
    wwh = np.array(np.mat(w.T) * wh)
    wa = np.array(np.mat(w.T) * a)
    h_new = h * (wa / wwh)

    ah = np.dot(a, h_new.T)
    wh = wh = np.dot(w, h_new)
    whh = np.array(wh * np.mat(h_new.T))
    w_new = w * (ah / whh)
    return w_new, h_new


def compute_distance(a, w, h):
    temp = np.array(np.dot(w, h))
    temp = a - temp
    temp *= temp
    return np.sum(temp)


def get_max_indices(w):
    ret = []
    for column in w.T:
        ret.append(column.argsort()[-3:][::-1])
    return ret


def run(min_delta, max_iter, k):
    a = tf_idf(read_term_document())
    terms = read_terms()
    print("Term-Document Matrix tf-idf normalised loaded...")
    w, h = init_wh(a, k)
    e = compute_distance(a, w, h)
    delta_e = e
    new_e = e
    smallest_e = e
    best_w = w
    i = 0
    while i < max_iter:  # at least 50 iterations, then wait until change in error gets very small
        if i > 3000:
            print("Max iterations reached!")
            break
        w, h = iter_step(a, w, h)
        new_e = compute_distance(a, w, h)
        if new_e < smallest_e:
            smallest_e = new_e
            print("[" + str(i) + "] New Best e: " + str(smallest_e))
            best_w = w
        delta_e = e - new_e
        print("[" + str(i) + "] Delta-e: " + str(delta_e))
        e = new_e
        i += 1
    print("Computation finished (Iterations=" + str(i) + ")! Error: " + str(e))
    for i in get_max_indices(best_w):
        print("Cluster " + str(i))
        for j in i:
            print("\tTerm: " + terms[j])


run(0.0000000001, 1000, 6)
