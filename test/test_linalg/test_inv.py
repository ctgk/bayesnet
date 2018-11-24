import unittest
import numpy as np
import bayesnet as bn


class TestInverse(unittest.TestCase):

    def test_inverse(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        Ainv = np.linalg.inv(A)
        self.assertTrue((Ainv == bn.linalg.inv(A).value).all())

        B = np.array([
            [-1., 1.],
            [1., 0.5]
        ])
        A = bn.Parameter(np.array([
            [-0.4, 0.7],
            [0.7, 0.7]
        ]))
        for _ in range(100):
            A.cleargrad()
            Ainv = bn.linalg.inv(A)
            Ainv.backward(2 * (Ainv.value - B))
            A.value -= 0.1 * A.grad

        self.assertTrue(np.allclose(A.value, np.linalg.inv(B)))


if __name__ == '__main__':
    unittest.main()
