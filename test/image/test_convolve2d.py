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

        p = bn.Parameter(kernel)
        output = bn.convolve2d(img, p, 2, 1)
        loss = bn.sum(bn.square(output))
        loss.backward()
        grad_backprop = p.grad
        grad_numerical = np.zeros_like(grad_backprop)
        eps = 1e-8
        for i, j in itertools.product(range(3), repeat=2):
            e = np.zeros_like(kernel)
            e[i, j] += eps
            loss_p = bn.sum(bn.square(bn.convolve2d(img, kernel + e, 2, 1))).value
            loss_m = bn.sum(bn.square(bn.convolve2d(img, kernel - e, 2, 1))).value
            grad_numerical[i, j] = (loss_p - loss_m) / (2 * eps)

        self.assertTrue(np.allclose(grad_backprop, grad_numerical))


if __name__ == '__main__':
    unittest.main()
