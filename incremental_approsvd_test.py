from incremental_approsvd import *
import numpy as np
import unittest

class TestIncrementalSVD(unittest.TestCase):

  def test_svd(self):
    k = 5
    n1 = 10
    n2 = 5

    mat_b1 = np.random.randn(k, n1)
    mat_b2 = np.random.randn(k, n2)

    nnz_b1 = np.count_nonzero(mat_b1)
    p1 = np.zeros(n1)
    for i in range(n1):
      p1[i] = np.count_nonzero(mat_b1[:, i]) / float(nnz_b1)

    nnz_b2 = np.count_nonzero(mat_b2)
    p2 = np.zeros(n2)
    for i in range(n2):
      p2[i] = np.count_nonzero(mat_b2[:, i]) / float(nnz_b2)

    mat_hk = incrementalApproSVD(mat_b1, mat_b2, n1-5, n2-1, k, p1, p2)
    mat_orig = np.hstack((mat_b1, mat_b2))
    np.testing.assert_array_almost_equal(np.dot(np.dot(mat_hk, mat_hk.T), mat_orig), mat_orig)

if __name__ == '__main__':
  unittest.main()
