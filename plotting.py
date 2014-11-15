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
    #for c in range(2, 7):
    term_indices, iterations, best_w, e = nmf.nmf(a, abort_error, max_iter, 6, num_terms)
    #    errors.append(e)

    plotter(best_w)


def plot_error(errors):
    fig = plt.figure()
    fig.suptitle('Euclidian Distance', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost function')
    i = 2
    hndls = []
    for e in errors:
        hndls.append(ax.plot(e, label=str(i) + ' Cluster')[0])
        i += 1
    plt.legend(handles=hndls)
    plt.show()


def plot_w(best_w):
    fig = plt.figure()
    fig.suptitle('Basis Matrix W', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Terms')
    ax.set_ylabel('Weight')
    T = range(best_w.shape[0])
    for i in range(best_w.shape[1]):
        plt.plot(T, best_w[:, i])
    plt.show()

plot_run(plot_w)

