import unittest
import numpy as np
import bayesnet as bn


class TestSoftplus(unittest.TestCase):

    def test_softplus(self):
        self.assertEqual(bn.softplus(0).value, np.log(2))


if __name__ == '__main__':
    unittest.main()
