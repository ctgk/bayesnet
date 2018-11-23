import unittest
import numpy as np
from bayesnet import Tensor


class TestTensor(unittest.TestCase):

    def test_tensor(self):
        values = (1, 1., np.array(0), np.ones(2), np.zeros((5, 4)))
        reprs = (
            "Tensor(value=1)",
            "Tensor(value=1.0)",
            f"Tensor(shape=(), dtype={np.array(0).dtype})",
            f"Tensor(shape=(2,), dtype={np.ones(2).dtype})",
            f"Tensor(shape=(5, 4), dtype={np.zeros((5, 4)).dtype})"
        )
        ndims = (0, 0, 0, 1, 2)
        shapes = ((), (), (), (2,), (5, 4))
        sizes = (1, 1, 1, 2, 20)

        for v, r, n, sh, si in zip(values, reprs, ndims, shapes, sizes):
            t = Tensor(v)
            self.assertIs(t.parent, None)
            self.assertEqual(repr(t), r)
            self.assertEqual(t.ndim, n)
            self.assertEqual(t.shape, sh)
            self.assertEqual(t.size, si)

        self.assertRaises(TypeError, Tensor, "abc")

    def test_backward(self):
        t = Tensor(1.)
        self.assertRaises(ValueError, t.backward, np.ones(1))
        self.assertRaises(TypeError, t.backward, "abc")

        t = Tensor(np.ones((2, 3)))
        self.assertRaises(ValueError, t.backward, 1)
        self.assertRaises(ValueError, t.backward, np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
