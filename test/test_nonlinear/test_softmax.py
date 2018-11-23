import unittest
import numpy as np
import bayesnet as bn


class TestSoftmax(unittest.TestCase):

    def test_softmax(self):
        self.assertTrue(np.allclose(bn.softmax(np.ones(4)).value, 0.25))


if __name__ == '__main__':
    unittest.main()
