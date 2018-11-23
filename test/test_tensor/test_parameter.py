import unittest
import numpy as np
from bayesnet import Parameter


class TestParameter(unittest.TestCase):

    def test_parameter(self):
        values = (1, np.ones((3, 4)))
        deltas = (2, np.ones((3, 4)))

        for v, d in zip(values, deltas):
            p = Parameter(v)
            self.assertIs(p.grad, None)
            p.backward(delta=d)
            if isinstance(d, np.ndarray):
                self.assertTrue((p.grad == d).all())
            else:
                self.assertEqual(p.grad, d)
            p.cleargrad()
            self.assertIs(p.grad, None)


if __name__ == '__main__':
    unittest.main()
