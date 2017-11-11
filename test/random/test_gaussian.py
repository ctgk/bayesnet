import unittest
import numpy as np
import bayesnet as bn


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        self.assertRaises(ValueError, bn.random.Gaussian, 0, -1)
        self.assertRaises(ValueError, bn.random.Gaussian, 0, np.array([1, -1]))


if __name__ == '__main__':
    unittest.main()
