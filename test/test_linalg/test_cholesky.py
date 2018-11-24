import unittest
import numpy as np
import bayesnet as bn


class TestCholesky(unittest.TestCase):

    def test_cholesky(self):
        A = np.array([
            [2., -1],
            [-1., 5.]
        ])
        L_expected = np.linalg.cholesky(A)
        Ap = bn.Parameter(A)
        L_actual = bn.linalg.cholesky(Ap).value
        self.assertTrue((L_expected == L_actual).all())

        T = np.array([
            [1., 0.],
            [-1., 2.]
        ])
        for _ in range(1000):
            Ap.cleargrad()
            L = bn.linalg.cholesky(Ap)
            L.backward(2 * (L.value - T))
            Ap.value -= 0.1 * Ap.grad

        self.assertTrue(np.allclose(Ap.value, T @ T.T))


if __name__ == '__main__':
    unittest.main()
