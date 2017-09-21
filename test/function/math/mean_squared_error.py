import unittest
import numpy as np
from bayes.tensor import Parameter
from bayes.function import mean_squared_error


class TestMeanSquaredError(unittest.TestCase):

    def test_mean_squared_error(self):
        x = np.random.rand(10, 3)
        y = np.random.rand(3)
        yp = Parameter(y)
        z = mean_squared_error(x, yp)
        self.assertEqual(z.value, 0.5 * np.square(x - y).mean())
        z.backward()
        self.assertTrue(np.allclose(yp.grad, (y - x).sum(axis=0) / 30))
