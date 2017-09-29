import unittest
import numpy as np
import bayesnet as bn


class TestDeterminant(unittest.TestCase):

    def test_determinant(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        detA = np.linalg.det(A)
        self.assertTrue((detA == bn.linalg.det(A).value).all())

        A = bn.Parameter(A)
        for _ in range(100):
            A.cleargrad()
            detA = bn.linalg.det(A)
            loss = bn.square(detA - 1)
            loss.backward()
            A.value -= 0.1 * A.grad
        self.assertAlmostEqual(detA.value, 1.)


if __name__ == '__main__':
    unittest.main()
