import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Trace(Function):

    def _forward(self, x):
        self._assert_ndim_equal_to(x, 2)
        return np.trace(x)

    def backward(self, delta):
        x = self.args[0]
        dx = np.eye(x.shape[0], x.shape[1]) * delta
        x.backward(dx)


def trace(x):
    return Trace().forward(x)
