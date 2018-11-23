import unittest
import numpy as np
import bayesnet as bn


class TestNanSum(unittest.TestCase):

    def test_nansum(self):
        x = np.random.rand(5, 1, 2)
        x[0, 0, 0] = np.nan
        xp = bn.Parameter(x)
        z = bn.nansum(xp)
        self.assertEqual(z.value, np.nansum(x))
        z.backward()
        g = np.ones((5, 1, 2))
        g[0, 0, 0] = 0
        self.assertTrue((xp.grad == g).all())


if __name__ == '__main__':
    unittest.main()
