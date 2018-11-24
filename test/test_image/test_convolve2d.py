import itertools
import unittest
import numpy as np
from scipy.ndimage.filters import correlate
import bayesnet as bn


class TestConvolve2d(unittest.TestCase):

    def test_convolve2d(self):
        np.random.seed(1234)

        img = np.random.randn(1, 5, 5, 1)
        kernel = np.random.randn(3, 3, 1, 1)
        output = bn.convolve2d(img, kernel)
        self.assertTrue(
            np.allclose(
                output.value[0, ..., 0],
                correlate(img[0, ..., 0], kernel[..., 0, 0])[1:-1, 1:-1]
            )
        )

        x = bn.Parameter(img)
        p = bn.Parameter(kernel)
        for _ in range(1000):
            x.cleargrad()
            p.cleargrad()
            output = bn.convolve2d(x, p, 2, 1)
            output.backward(2 * (output.value - 1))
            x.value -= x.grad * 0.01
            p.value -= p.grad * 0.01
        self.assertTrue(np.allclose(bn.convolve2d(x, p, 2, 1).value, 1))


if __name__ == '__main__':
    unittest.main()
