import unittest
import numpy as np
import bayesnet as bn


class TestSolve(unittest.TestCase):

    def test_solve(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        B = np.array([1., 2.])[:, None]
        AinvB = np.linalg.solve(A, B)
        self.assertTrue((AinvB == bn.linalg.solve(A, B).value).all())

        A = bn.Parameter(A)
        B = bn.Parameter(B)
        for _ in range(100):
            A.cleargrad()
            B.cleargrad()
            AinvB = bn.linalg.solve(A, B)
            loss = bn.square(AinvB - 1).sum()
            loss.backward()
            A.value -= A.grad
            B.value -= B.grad
        self.assertTrue(np.allclose(AinvB.value, 1))


if __name__ == '__main__':
    unittest.main()
