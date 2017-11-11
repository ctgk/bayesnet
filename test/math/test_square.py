import unittest
import numpy as np
import bayesnet as bn


class TestSqrt(unittest.TestCase):

    def test_sqrt(self):
        x = bn.Parameter(2.)
        y = bn.square(x)
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4)


if __name__ == '__main__':
    unittest.main()
