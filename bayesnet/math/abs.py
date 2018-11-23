import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Abs(Function):

    def _forward(self, x):
        self.sign = np.sign(x.value)
        return np.abs(x.value)

    def backward(self, delta):
        dx = self.sign * delta
        self.args[0].backward(dx)


def abs(x):
    """
    element-wise absolute function
    """
    return Abs().forward(x)
