import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Abs(Function):

    def _forward(self, x):
        self.sign = np.sign(x)
        return np.abs(x)

    def _backward(self, delta, x):
        dx = self.sign * delta
        return dx


def abs(x):
    """
    element-wise absolute function
    """
    return Abs().forward(x)
