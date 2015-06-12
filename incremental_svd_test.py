from incremental_svd import *
import numpy as np
import unittest

class TestIncrementalSVD(unittest.TestCase):

  def test_svd(self):
    mat_a1 = np.random.randn(2, 3)
    mat_a2 = np.random.randn(2, 1)
    k = 2 # same as the original
    mat_u, mat_s, mat_vt = incrementalSVD(mat_a1, mat_a2, k)
    np.testing.assert_array_almost_equal(np.dot(np.dot(mat_u, mat_s), mat_vt), np.hstack((mat_a1, mat_a2)))

  def test_invalid_k(self):
    self.assertRaises(ValueError, incrementalSVD, np.random.randn(2, 3), np.random.randn(2, 1), 0)
    self.assertRaises(ValueError, incrementalSVD, np.random.randn(2, 3), np.random.randn(2, 1), 100)

  def test_different_m(self):
    self.assertRaises(ValueError, incrementalSVD, np.random.randn(2, 3), np.random.randn(3, 1), 2)

if __name__ == '__main__':
  unittest.main()
