import unittest
import numpy as np
from bayes.tensor import Parameter
from bayes.function import square


class TestSqrt(unittest.TestCase):

    def test_sqrt(self):
        x = Parameter(2.)
        y = square(x)
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4)
