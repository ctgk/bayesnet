import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Exp(Function):

    def _forward(self, x):
        self.output = np.exp(x)
        return self.output

    def backward(self, delta):
        dx = self.output * delta
        self.args[0].backward(dx)


def exp(x):
    """
    element-wise exponential function
    """
    return Exp().forward(x)
