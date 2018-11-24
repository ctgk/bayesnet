import unittest
import numpy as np
import bayesnet as bn


class TestTrace(unittest.TestCase):

    def test_trace(self):
        arrays = [
            np.random.normal(size=(2, 2)),
            np.random.normal(size=(3, 4))
        ]

        for arr in arrays:
            arr = bn.Parameter(arr)
            tr_arr = bn.linalg.trace(arr)
            self.assertEqual(tr_arr.value, np.trace(arr.value))

        a = np.array([
            [1.5, 0],
            [-0.1, 1.1]
        ])
        a = bn.Parameter(a)
        for _ in range(100):
            a.cleargrad()
            t = bn.linalg.trace(a)
            t.backward(2 * (t.value - 2))
            a.value -= 0.1 * a.grad
        self.assertEqual(bn.linalg.trace(a).value, 2)


if __name__ == '__main__':
    unittest.main()
