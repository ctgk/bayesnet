import unittest
import numpy as np
import bayesnet as bn


class TestLogdet(unittest.TestCase):

    def test_logdet(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        logdetA = np.linalg.slogdet(A)[1]
        self.assertTrue((logdetA == bn.linalg.logdet(A).value).all())

        A = bn.Parameter(A)
        for _ in range(100):
            A.cleargrad()
            logdetA = bn.linalg.logdet(A)
            loss = bn.square(logdetA - 1)
            loss.backward()
            A.value -= 0.1 * A.grad
        self.assertAlmostEqual(logdetA.value, 1)


if __name__ == '__main__':
    unittest.main()
