import unittest
import numpy as np
import bayesnet as bn


class TestProduct(unittest.TestCase):

    def test_product(self):
        arrays = [
            1,
            np.arange(1, 5),
            np.arange(1, 7).reshape(2, 3),
            np.arange(1, 7).reshape(2, 3, 1)
        ]
        axes = [
            None,
            None,
            1,
            (0, 2)
        ]
        keepdims = [
            False,
            False,
            True,
            False
        ]
        grads = [
            1,
            np.array([24., 12., 8., 6.]),
            np.array([
                [6., 3., 2.],
                [30., 24., 20.]
            ]),
            np.array([4., 5., 6., 1., 2., 3.]).reshape(2, 3, 1)
        ]

        for arr, ax, keep, g in zip(arrays, axes, keepdims, grads):
            a = bn.Parameter(arr)
            b = a.prod(ax, keep)
            b.backward(np.ones(b.shape))
            if isinstance(g, int):
                self.assertEqual(g, a.grad)
            else:
                self.assertTrue((g == a.grad).all())


if __name__ == '__main__':
    unittest.main()
