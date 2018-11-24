import unittest
import itertools
import numpy as np
from scipy.ndimage.filters import correlate
import bayesnet as bn


class TestDeconvolve2d(unittest.TestCase):

    def test_deconvolve2d(self):
        np.random.seed(1234)

        img = np.random.randn(1, 5, 5, 1)
        kernel = np.random.randn(3, 3, 1, 1)
        output = bn.deconvolve2d(img, kernel)
        self.assertTrue(
            np.allclose(
                output.value[0, 1:-1, 1:-1, 0],
                correlate(
                    img[0, :, :, 0],
                    kernel[::-1, ::-1, 0, 0],
                    mode="constant"
                )
            )
        )

        x = bn.Parameter(img)
        p = bn.Parameter(kernel)
        for _ in range(1000):
            x.cleargrad()
            p.cleargrad()
            output = bn.deconvolve2d(x, p)
            output.backward(2 * (output.value - 0))
            x.value -= x.grad * 0.01
            p.value -= p.grad * 0.01
        self.assertTrue(np.allclose(bn.deconvolve2d(x, p).value, 0))


if __name__ == '__main__':
    unittest.main()
