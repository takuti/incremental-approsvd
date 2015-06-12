from incremental_svd import *
import numpy as np
import numpy.linalg as ln
import sys

def incrementalApproSVD(mat_b1, mat_b2, c1, c2, k, p1, p2):
  """Apply Incremental ApproSVD for a matrix with new columns

  :param mat_b1: original matrix (m x n1)
  :param mat_b2: new columns (m x n2)
  :param c1: the number of sampled columns from B1
  :param c2: the number of sampled columns from B2
  :param k: rank-k for the approximated result
  :param p1: sampling probabilities for each column in B1
  :param p2: sampling probabilities for each column in B2
  :returns: H_k as an output of Incremental ApproSVD (H_k H_k^T = I)
  """

  if mat_b1.shape[0] != mat_b2.shape[0]:
    raise ValueError('Error: the number of rows both in mat_a1 and mat_a2 should be the same')

  if len(p1[p1<0]) != 0:
    raise ValueError('Error: negative probabilities in p1 are not allowed')
  if len(p2[p2<0]) != 0:
    raise ValueError('Error: negative probabilities in p2 are not allowed')

  if not np.isclose(sum(p1), 1.):
    raise ValueError('Error: sum of the probabilities must be 1 for p1')
  if not np.isclose(sum(p2), 1.):
    raise ValueError('Error: sum of the probabilities must be 1 for p2')

  # get the number of rows and columns
  m = mat_b1.shape[0]
  n1 = mat_b1.shape[1]
  n2 = mat_b2.shape[1]

  if c1 >= n1:
    raise ValueError('Error: c1 must be less than n1')
  if c2 >= n2:
    raise ValueError('Error: c2 must be less than n2')

  if k < 1:
    raise ValueError('Error: rank k must be greater than or equal to 1')
  if k > min(m, c1 + c2):
    raise ValueError('Error: rank k must be less than or equal to min(m, n1 + n2)')

  # sample c1 columns from B1, and combine them as a matrix C1
  mat_c1 = np.zeros((m, c1))
  samples = np.random.choice(range(n1), c1, replace=False, p=p1)
  for t in range(c1):
    mat_c1[:, t] = mat_b1[:, samples[t]] / np.sqrt(c1 * p1[samples[t]])

  # sample c2 columns from B2, and combine them as a matrix C2
  mat_c2 = np.zeros((m, c2))
  samples = np.random.choice(range(n2), c2, replace=False, p=p2)
  for t in range(c2):
    mat_c2[:, t] = mat_b2[:, samples[t]] / np.sqrt(c2 * p2[samples[t]])

  # apply Incremental SVD for smaller matrices C1, C2, and get only U_k as H_k
  return incrementalSVD(mat_c1, mat_c2, k, True)
