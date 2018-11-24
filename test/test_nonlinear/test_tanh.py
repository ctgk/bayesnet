import unittest
import numpy as np
import bayesnet as bn


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        x = bn.Parameter(2)
        y = bn.tanh(x)
        self.assertEqual(y.value, np.tanh(2))
        y.backward()
        self.assertEqual(x.grad, 1 - np.tanh(2) ** 2)


if __name__ == '__main__':
    unittest.main()
