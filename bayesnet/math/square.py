import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Square(Function):
    """
    element-wise square of the input
    y = x * x
    """

    @staticmethod
    def _forward(x):
        return np.square(x)

    @staticmethod
    def _backward(delta, x):
        dx = 2 * x * delta
        return dx


def square(x):
    """
    element-wise square of the input
    y = x * x
    """
    return Square().forward(x)
