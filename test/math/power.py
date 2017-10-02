import unittest
import numpy as np
import bayesnet as bn


class TestPower(unittest.TestCase):

    def test_power(self):
        x = bn.Parameter(2.)
        y = 2 ** x
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4 * np.log(2))

        x = np.random.rand(10, 2)
        xp = bn.Parameter(x)
        y = xp ** 3
        self.assertTrue((y.value == x ** 3).all())
        y.backward(np.ones((10, 2)))
        self.assertTrue((xp.grad == 3 * x ** 2).all())


if __name__ == '__main__':
    unittest.main()
