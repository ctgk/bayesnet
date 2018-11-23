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
        return np.square(x.value)

    def backward(self, delta):
        x = self.args[0]
        dx = 2 * x.value * delta
        x.backward(dx)


def square(x):
    """
    element-wise square of the input
    y = x * x
    """
    return Square().forward(x)
