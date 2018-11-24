import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Exp(Function):

    def _forward(self, x):
        self.output = np.exp(x)
        return self.output

    def _backward(self, delta, x):
        dx = self.output * delta
        return dx


def exp(x):
    """
    element-wise exponential function
    """
    return Exp().forward(x)
