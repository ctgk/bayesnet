import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Log(Function):
    """
    element-wise natural logarithm of the input
    y = log(x)
    """

    @staticmethod
    def _forward(x):
        return np.log(x.value)

    def backward(self, delta):
        x = self.args[0]
        dx = delta / x.value
        x.backward(dx)


def log(x):
    """
    element-wise natural logarithm of the input
    y = log(x)
    """
    return Log().forward(x)
