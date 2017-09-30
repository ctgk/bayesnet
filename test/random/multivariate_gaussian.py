import unittest
import numpy as np
import bayesnet as bn


class TestMultivariateGaussian(unittest.TestCase):

    def test_multivariate_gaussian(self):
        self.assertRaises(ValueError, bn.random.MultivariateGaussian, np.zeros(2), np.eye(3))
        self.assertRaises(ValueError, bn.random.MultivariateGaussian, np.zeros(2), np.eye(2) * -1)

        g = bn.random.MultivariateGaussian(
            np.ones(2), np.eye(2) * 2
        )
        self.assertEqual(g.KLqp(g).value, 0)


if __name__ == '__main__':
    unittest.main()
