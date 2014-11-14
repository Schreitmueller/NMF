__author__ = 'Philipp'

import matplotlib.pyplot as plt

import nmf as nmf


def plot_run(plotter):
    a = nmf.tf_idf(nmf.read_term_document())
    terms = nmf.read_terms()
    abort_error = 1e-7
    max_iter = 500
    num_terms = 3
    errors = []
    for c in range(2, 4):
        term_indices, iterations, best_w, e = nmf.nmf(a, abort_error, max_iter, c, num_terms)
        errors.append(e)

    plotter(errors, best_w)


def plot_error(errors):
    fig = plt.figure()
    fig.suptitle('Euclidian Distance', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('axes title')

    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    plt.show()
# T = range(best_w.shape[0])

# for i in range(best_w.shape[1]):
# plt.plot(T, best_w[:, i])

plot_run(plot_error)

