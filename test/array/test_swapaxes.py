import unittest
import numpy as np
import bayesnet as bn


class TestSwapaxes(unittest.TestCase):

    def test_swapaxes(self):
        arrays = [
            np.random.normal(size=(2, 3)),
            np.random.normal(size=(2, 3, 4))
        ]
        axes = [
            (0, 1),
            (-1, -2)
        ]

        for arr, ax in zip(arrays, axes):
            arr = bn.Parameter(arr)
            arr_swapped = bn.swapaxes(arr, ax[0], ax[1])
            self.assertEqual(arr_swapped.shape, np.swapaxes(arr.value, ax[0], ax[1]).shape)
            da = np.random.normal(size=arr_swapped.shape)
            arr_swapped.backward(da)
            self.assertEqual(arr.grad.shape, arr.shape)


if __name__ == '__main__':
    unittest.main()
