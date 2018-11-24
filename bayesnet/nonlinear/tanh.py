import numpy as np
from bayesnet.tensor.constant import Constant
from bayesnet.tensor.tensor import Tensor
from bayesnet.function import Function


class Tanh(Function):

    def _forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def _backward(self, delta, x):
        dx = (1 - np.square(self.output)) * delta
        return dx


def tanh(x):
    """
    hyperbolic tangent function
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return Tanh().forward(x)
