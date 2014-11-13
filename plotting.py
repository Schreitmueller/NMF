__author__ = 'Philipp'

import matplotlib.pyplot as plt

import nmf as nmf


a = nmf.tf_idf(nmf.read_term_document())
terms = nmf.read_terms()
abort_error = 1e-6
max_iter = 500
num_terms = 3
best_terms, iterations, best_w = nmf.nmf(a, abort_error, max_iter, 6, 3)

T = range(best_w.shape[0])

for i in range(best_w.shape[1]):
    plt.plot(T, best_w[:, i])

plt.show()

