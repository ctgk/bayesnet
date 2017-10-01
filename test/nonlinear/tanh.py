import unittest
import bayesnet as bn


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        self.assertEqual(bn.tanh(0).value, 0)


if __name__ == '__main__':
    unittest.main()
