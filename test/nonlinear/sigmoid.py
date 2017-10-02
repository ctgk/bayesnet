import unittest
import bayesnet as bn


class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(bn.sigmoid(0).value, 0.5)


if __name__ == '__main__':
    unittest.main()
