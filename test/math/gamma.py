import unittest
import bayesnet as bn


class TestGamma(unittest.TestCase):

    def test_gamma(self):
        self.assertEqual(24, bn.gamma(5).value)
        a = bn.Parameter(2.5)
        eps = 1e-5
        b = bn.gamma(a)
        b.backward()
        num_grad = ((bn.gamma(a + eps) - bn.gamma(a - eps)) / (2 * eps)).value
        self.assertAlmostEqual(a.grad, num_grad)


if __name__ == '__main__':
    unittest.main()
