import unittest
import numpy as np
import bayesnet as bn


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        self.assertRaises(ValueError, bn.random.Gaussian, 0, -1)
        self.assertRaises(ValueError, bn.random.Gaussian, 0, np.array([1, -1]))

        g = bn.random.Gaussian(0, 1)
        variables = [
            bn.random.Gaussian(0, 1),
            bn.random.Gaussian(-4, 1),
            bn.random.Gaussian(2, 1)
        ]
        divergences = [0, 8, 2]
        for v, d in zip(variables, divergences):
            self.assertEqual(g.KLqp(v).value, d)


if __name__ == '__main__':
    unittest.main()
