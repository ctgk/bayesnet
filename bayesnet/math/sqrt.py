import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Sqrt(Function):
    """
    element-wise square root of the input
    y = sqrt(x)
    """

    def _forward(self, x):
        self.output = np.sqrt(x)
        return self.output

    def backward(self, delta):
        dx = 0.5 * delta / self.output
        self.args[0].backward(dx)


def sqrt(x):
    """
    element-wise square root of the input
    y = sqrt(x)
    """
    return Sqrt().forward(x)
