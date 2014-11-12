__author__ = 'Philipp Schreitmueller'

import math as m

import numpy as np

np.random.seed(1)


def read_term_document():
    """
    Reads the 'bbcnews.mtx' file and transforms it in a word-term array
    :return: term-document array
    """
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
    """
    Reads the bbcnes.terms file in a list
    :return: List with all available terms
    """
    f = open("data/bbcnews.terms")
    ret = []
    for line in f.readlines():
        ret.append(line.replace("\n", ""))
    f.close()
    return ret


def tf_idf(a):
    """
    Applies normalised tf_idf on the raw term-document matrix
    :param a: Term-document array
    :return: tf_idf term-document array
    """
    sumTerms = [sum(colum) for colum in a.T]  # sum of all terms in a document
    sumWord = [sum(x >= 1 for x in row) for row in a]
    numDoc = len(a)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i, j] = (a[i, j] / sumTerms[j]) * m.log(numDoc / sumWord[i])
    return a


def init_wh(a, k):
    """
    Initialises w,h arrays which product approximates the term-document matrix
    :param a: normalised term-document array
    :param k: number of clusters
    :return: tuple of arrays w and h
    """
    w = np.random.random(a.shape[0] * k).reshape(a.shape[0], k) * np.average(a)
    h = np.random.random(k * a.shape[1]).reshape(k, a.shape[1]) * np.average(a)
    return w, h


def iter_step(a, w, h):
    """
    Performs a refinition of matrices w and h
    :param a: normalised term-document array
    :param w: array w
    :param h: array h
    :return: tuple of new w and h
    """
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
    """
    Computes euclidian distance between original array and the product of w and h
    :param a: normalised term-document array
    :param w: array
    :param h: array
    :return: distance value
    """
    temp = np.array(np.dot(w, h))
    temp = a - temp
    temp *= temp
    return np.sum(temp)


def get_max_indices(w, terms):
    """
    Returns the indices of the most important terms per cluster
    :param w: array
    :param terms: number of terms to be returned per cluster
    :return: list of lists with indices of most common terms per cluster
    """
    ret = []
    for column in w.T:
        ret.append(column.argsort()[-terms:][::-1])
    return ret


def run(min_delta, max_iter, k, num_terms):
    """
    The main function: performs NMF clustering with euclidian distance as cost function
    with two abort criteria
    :param min_delta: Stop loop if change in error is smaller than this value
    :param max_iter: Stop loop after this number of loop executions
    :param k: Number of cluster
    :param num_terms: Number of terms to be listed per clusters
    :return:
    """
    print("Clustering into " + str(k) + " Clusters. Find smallest error within " + str(max_iter) + " Iterations or")
    print("abort when change in error is smaller than " + str(min_delta))
    a = tf_idf(read_term_document())
    terms = read_terms()
    print("Term-Document Matrix tf-idf normalised loaded...")
    w, h = init_wh(a, k)
    best_w = w
    e = delta_e = new_e = smallest_e = compute_distance(a, w, h)
    i = 0
    while i < max_iter and delta_e > min_delta:
        w, h = iter_step(a, w, h)
        new_e = compute_distance(a, w, h)
        if new_e < smallest_e:
            smallest_e = new_e
            #print("[" + str(i) + "] New Best e: " + str(smallest_e)) # DEBUG
            best_w = w
        delta_e = e - new_e
        e = new_e
        i += 1
    print("Computation finished (Iterations=" + str(i) + ")! Error: " + str(smallest_e))
    for i in get_max_indices(best_w, num_terms):
        print("Cluster " + str(i))
        for j in i:
            print("\tTerm: " + terms[j])


abort_error = 1e-6
max_iter = 500
num_terms = 3
for c in range(2, 6):
    run(abort_error, max_iter, c, num_terms)
