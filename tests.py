__author__ = 'Philipp Schreitmueller'

import numpy as np
import nmf as nmf

def get_test_a():
    """
    Test method
    :return: Sample Term-Document matrix
    """
    return np.array(((3.0, 4.0, 0), (17.0, 0.0, 1.0), (0.0, 2.0, 3.0)))


def tests():
    """
    Testing the implemented functions with small values to ensure functinoality
    :return: null
    """
    testmatrix = get_test_a()
    print(testmatrix)
    print(nmf.tf_idf(testmatrix))
    print(nmf.iter_step(get_test_a(), np.array(((1, 1), (1, 1), (1, 1))), np.array(((2, 2, 2), (2, 2, 2)))))
    print(nmf.compute_distance(np.array(((5, 5, 5), (5, 5, 5), (5, 5, 5))), np.array(((1, 1), (1, 1), (1, 1))),
                               np.array(((4, 4, 4), (4, 4, 4)))))
    a = np.array(((2, 1, 5), (4, 6, 2), (6, 5, 4)))
    b = []
    for colum in a.T:
        b.append(colum.argsort()[-3:][::-1])
    print(b)

tests()


