import unittest
import numpy as np
import bayesnet as bn


class TestCholesky(unittest.TestCase):

    def test_cholesky(self):
        A = np.array([
            [2., -1],
            [-1., 5.]
        ])
        L = np.linalg.cholesky(A)
        Ap = bn.Parameter(A)
        L_test = bn.linalg.cholesky(Ap)
        self.assertTrue((L == L_test.value).all())

        T = np.array([
            [1., 0.],
            [-1., 2.]
        ])
        for _ in range(1000):
            Ap.cleargrad()
            L_ = bn.linalg.cholesky(Ap)
            loss = bn.square(T - L_).sum()
            loss.backward()
            Ap.value -= 0.1 * Ap.grad

        self.assertTrue(np.allclose(Ap.value, T @ T.T))


if __name__ == '__main__':
    unittest.main()
