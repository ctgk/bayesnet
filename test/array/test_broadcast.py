import unittest
import numpy as np
import bayesnet as bn
from bayesnet.array.broadcast import broadcast_to
from bayesnet.array.broadcast import broadcast


class TestBroadcastTo(unittest.TestCase):

    def test_broadcast_to(self):
        x = bn.Parameter(np.ones((1, 1)))
        shape = (5, 2, 3)
        y = broadcast_to(x, shape)
        self.assertEqual(y.shape, shape)
        y.backward(np.ones(shape))
        self.assertTrue((x.grad == np.ones((1, 1)) * 30).all())

    def test_broadcast(self):
        x = bn.Parameter(np.ones((2, 1, 3)))
        y = bn.Parameter(np.ones((4, 3)))
        xb, yb = broadcast([x, y])
        self.assertEqual(xb.shape, (2, 4, 3))
        self.assertEqual(yb.shape, (2, 4, 3))
        xb.backward(np.ones((2, 4, 3)))
        yb.backward(np.ones((2, 4, 3)))
        self.assertTrue((x.grad == np.ones((2, 1, 3)) * 4).all())
        self.assertTrue((y.grad == np.ones((4, 3)) * 2).all())

if __name__ == '__main__':
    unittest.main()
