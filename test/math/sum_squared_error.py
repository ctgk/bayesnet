import unittest
import numpy as np
import bayesnet as bn


class TestSumSquaredError(unittest.TestCase):

    def test_sum_squared_error(self):
        x = np.random.rand(10, 3)
        y = np.random.rand(3)
        yp = bn.Parameter(y)
        z = bn.sum_squared_error(x, yp)
        self.assertEqual(z.value, 0.5 * np.square(x - y).sum())
        z.backward()
        self.assertTrue((yp.grad == (y - x).sum(axis=0)).all())


if __name__ == '__main__':
    unittest.main()
